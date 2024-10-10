import numpy as np
import torch
import gc
import time


class GreedySearch:
    def __init__(self, model, tokenizer, model_type, em_splitter=None
                , compute_ppl=False):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.past_kv = None
        self.compute_ppl = compute_ppl
        self.em_splitter = em_splitter

    def clear(self):
        self.past_kv = None
        gc.collect()

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs

    def generate(self, text=None, input_ids=None, em_labels=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, em_labels, **kwargs)

        return result

    def _random_splitter(self, ids):
        where2split = torch.zeros_like(ids)

        for i, elem_ids in enumerate(ids):
            splits = np.random.randint(low=0, high=len(elem_ids)-1, size=10)  # Make sure you don't split on the last token
            where2split[i][0] = 1
            where2split[i][splits] = 1
        return where2split.bool()
    
    def _model_pass(self, input_ids, attention_mask, past_key_values, em_labels=None, labels=None):
        if not self.compute_ppl:
            labels = None

        if self.model_type == "em-llm":
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
                em_labels=em_labels,
                labels=labels,
                output_attentions=True,
            )
        else:
            raise NotImplementedError

        return out

    def _decode(self, input_ids, em_labels=None, max_length=100, extra_end_token_ids=[]
                , chunk_size: int = 4096, output=False, **kwargs):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        if (em_labels is not None) and (em_labels.dim() == 1):
            em_labels = em_labels[None, :]
        input_ids = input_ids.cuda()
        print(f"Context Length: {input_ids.size()}")
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""

        total_loss = 0
        chunk_ppl = []
        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                avg_time = 0
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    start_time = time.time()
                    ed = min(input_ids.size(1) - 1, st + chunk_size)

                    if self.em_splitter == "surprisal":
                        em_input = input_ids[:, st+1: ed+1]
                    elif self.em_splitter == "random":
                        em_input = self._random_splitter(input_ids[:, st: ed])
                        assert em_input.dtype == torch.bool
                    elif self.em_splitter == "sentence":
                        em_input = em_labels[:, st: ed]
                        assert em_input.dtype == torch.bool
                    else:
                        em_input = None

                    if em_input is not None:
                        assert em_input.shape == input_ids[:, st: ed].shape, f"Shape mismatch in em_labels and input_ids"
                    
                    if past_key_values is not None: 
                        if past_key_values[0].allow_disk_offload is None and input_ids.size(1) > kwargs["disk_offload_threshold"]:
                            print(f"Inputs have length {input_ids.size(1)}: allowing disk offload for past_key_values.")
                            for pkv in past_key_values:
                                pkv.allow_disk_offload = True
                        elif past_key_values[0].vector_offload and input_ids.size(1) > kwargs["vector_offload_threshold"] and past_key_values[0].block_repr_k[0].data.device != torch.device('cpu'):
                            for pkv in past_key_values:
                                pkv._offload_vector()

                    out = self._model_pass(
                        input_ids=input_ids[:, st: ed],
                        attention_mask=attention_mask[:, :ed],
                        past_key_values=past_key_values,
                        labels=input_ids[:, st+1: ed+1],
                        em_labels=em_input,
                    )

                    logits, past_key_values = out.logits, out.past_key_values
                    
                    try:
                        loss = out.loss.detach().cpu() if out.loss is not None else None
                    except:
                        loss = None
                    if loss is not None:
                        ppl = torch.exp(loss).item() if self.compute_ppl else None
                        total_loss += loss * (ed - st)
                    else:
                        ppl = None
                    
                    chunk_ppl.append(ppl)

                    time_taken = round(time.time() - start_time, 2)
                    avg_time += time_taken
                    log = f"Chunk: {int(st / chunk_size + 1)}/{(input_ids.size(1))//chunk_size}, ppl: {ppl}, time: {time_taken}s"
                    print(log)
                    if int(st / chunk_size + 1) % 100 == 0:
                        print(torch.cuda.memory_summary())
                        print(f"Average time taken per chunk: {round(avg_time/int(st / chunk_size + 1), 2)}s")
                        gc.collect()
 
                total_loss /= ed
                out = self._model_pass(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                )
                logits, past_key_values = out.logits, out.past_key_values
            else:
                out = self._model_pass(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
           
            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if self.compute_ppl:
            chunk_ppl = chunk_ppl
            total_ppl = torch.exp(total_loss).item()
        else:
            chunk_ppl = None
            total_ppl = None          

        return {"pred": self.tokenizer.decode(input_ids.squeeze(0)[length:]), "chunk_ppl": chunk_ppl, "total_ppl": total_ppl}
  
