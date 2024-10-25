
<center>

<table>
<tr>
    <th><strong>Task</strong></th>
    <th><strong>RAG</strong></th>
    <th><strong>FC</strong></th>
    <th><strong>EM-LLM</strong></th>
</tr>
<tr>
    <td colspan="4" align="center"><strong>Long-Bench</strong></td>
</tr>
<tr>
    <td>NarrativeQA</td>
    <td>22.54</td>
    <td><strong>29.14</strong></td>
    <td>26.05</td>
</tr>
<tr>
    <td>Qasper</td>
    <td><strong>45.45</strong></td>
    <td>45.34</td>
    <td>44.41</td>
</tr>
<tr>
    <td>MultiFieldQA</td>
    <td>51.67</td>
    <td><strong>54.98</strong></td>
    <td>52.52</td>
</tr>
<tr>
    <td>HotpotQA</td>
    <td><strong>55.93</strong></td>
    <td>54.01</td>
    <td>54.02</td>
</tr>
<tr>
    <td>2WikiMQA</td>
    <td>42.93</td>
    <td><strong>45.95</strong></td>
    <td>45.72</td>
</tr>
<tr>
    <td>Musique</td>
    <td>30.90</td>
    <td><strong>33.52</strong></td>
    <td>25.37</td>
</tr>
<tr>
    <td>GovReport</td>
    <td>29.91</td>
    <td>34.49</td>
    <td><strong>35.04</strong></td>
</tr>
<tr>
    <td>QMSum</td>
    <td><strong>24.97</strong></td>
    <td>25.14</td>
    <td>24.31</td>
</tr>
<tr>
    <td>MultiNews</td>
    <td>26.77</td>
    <td>27.00</td>
    <td><strong>27.76</strong></td>
</tr>
<tr>
    <td>TREC</td>
    <td>22.50</td>
    <td>4.50</td>
    <td><strong>71.50</strong></td>
</tr>
<tr>
    <td>TriviaQA</td>
    <td>88.11</td>
    <td>89.07</td>
    <td><strong>92.34</strong></td>
</tr>
<tr>
    <td>SAMSum</td>
    <td>7.56</td>
    <td>8.68</td>
    <td><strong>43.31</strong></td>
</tr>
<tr>
    <td>PassageRetrieval</td>
    <td>65.50</td>
    <td><strong>100.00</strong></td>
    <td>99.50</td>
</tr>
<tr>
    <td>LCC</td>
    <td>13.16</td>
    <td>19.30</td>
    <td><strong>67.45</strong></td>
</tr>
<tr>
    <td>RepoBench-P</td>
    <td>18.66</td>
    <td>18.33</td>
    <td><strong>64.33</strong></td>
</tr>
<tr>
    <td><strong>Avg. score:</strong></td>
    <td>36.44</td>
    <td>39.30</td>
    <td><strong>51.58</strong></td>
</tr>
<!-- Separator Line -->
<tr>
    <td colspan="4" align="center"><strong>&infin;-Bench</strong></td>
</tr>
<tr>
    <td>Code.Debug</td>
    <td><strong>22.59</strong></td>
    <td>21.70</td>
    <td>22.59</td>
</tr>
<tr>
    <td>Math.Find</td>
    <td>35.43</td>
    <td>26.29</td>
    <td><strong>36.00</strong></td>
</tr>
<tr>
    <td>Retrieve.KV</td>
    <td>31.80</td>
    <td>92.60</td>
    <td><strong>96.80</strong></td>
</tr>
<tr>
    <td>En.MC</td>
    <td><strong>64.19</strong></td>
    <td>58.07</td>
    <td>44.54</td>
</tr>
<tr>
    <td>Retrieve.PassKey</td>
    <td>100.00</td>
    <td>100.00</td>
    <td>100.00</td>
</tr>
<tr>
    <td>Retrieve.Number</td>
    <td>99.83</td>
    <td>99.32</td>
    <td><strong>100.00</strong></td>
</tr>
<tr>
    <td><strong>Avg. score:</strong></td>
    <td>58.97</td>
    <td>66.33</td>
    <td><strong>66.66</strong></td>
</tr>
</table>

</center>

**Table 1:** EM-LLM<sub>S</sub> (4K+4K) vs. RAG (NV-Embed-v2 retriever) vs. full-context, with LLaMa-3.1-8B as the base LLM, evaluated on LongBench and $\infty$-Bench.

---

<center>

<table>
  <!-- First Row: Group Headers -->
  <tr>
    <th rowspan="2"><strong>Base LLM</strong></th>
    <th rowspan="2"><strong>Method</strong></th>
    <th colspan="7"><strong>Long-Bench Tasks</strong></th>
    <th colspan="6"><strong>&infin;-Bench Tasks</strong></th>
  </tr>
  <!-- Second Row: Individual Task Headers -->
  <tr>
    <th><strong>SQA</strong></th>
    <th><strong>MQA</strong></th>
    <th><strong>Sum</strong></th>
    <th><strong>FSL</strong></th>
    <th><strong>Ret</strong></th>
    <th><strong>Cod</strong></th>
    <th style="border-left:1px solid; border-right:3px double;"><strong>Avg.</strong></th>
    <th><strong>C.D</strong></th>
    <th><strong>M.F</strong></th>
    <th><strong>MC</strong></th>
    <th><strong>R.KV</strong></th>
    <th><strong>R.P</strong></th>
    <th><strong>R.N</strong></th>
  </tr>
  <!-- Subheader: 7-8B parameter models -->
  <tr>
    <td colspan="15" align="center"><strong>7-8B parameter models</strong></td>
  </tr>
  <!-- Mistral v2 Rows -->
  <tr>
    <td><strong>Mistral v2</strong></td>
    <td>InfLLM (4k+2k)</td>
    <td><strong>33</strong></td>
    <td>25.5</td>
    <td>27.1</td>
    <td>66.1</td>
    <td>64</td>
    <td>54.8</td>
    <td style="border-left:1px solid; border-right:3px double;">41.9</td>
    <td><strong>29.4</strong></td>
    <td>26.6</td>
    <td><strong>43.2</strong></td>
    <td>95.6</td>
    <td>100</td>
    <td>99.8</td>
  </tr>
  <tr>
    <td></td>
    <td>EM-LLM<sub>SM+C</sub></td>
    <td>32.9</td>
    <td><strong>27</strong></td>
    <td><strong>27.2</strong></td>
    <td><strong>66.8</strong></td>
    <td><strong>84.1</strong></td>
    <td><strong>54.8</strong></td>
    <td style="border-left:1px solid; border-right:3px double;"><strong>43.7</strong></td>
    <td>28.2</td>
    <td><strong>27.1</strong></td>
    <td>42.8</td>
    <td><strong>99</strong></td>
    <td>100</td>
    <td>99.8</td>
  </tr>
  <!-- LLaMA 3 Rows -->
  <tr>
    <td><strong>LLaMA 3</strong></td>
    <td>InfLLM (4k+4k)</td>
    <td>38.5</td>
    <td>36.9</td>
    <td>27</td>
    <td>69</td>
    <td>84</td>
    <td><strong>53.2</strong></td>
    <td style="border-left:1px solid; border-right:3px double;">47</td>
    <td>30.5</td>
    <td><strong>23.7</strong></td>
    <td><strong>43.7</strong></td>
    <td><strong>5</strong></td>
    <td>100</td>
    <td>99</td>
  </tr>
  <tr>
    <td></td>
    <td>EM-LLM<sub>S</sub></td>
    <td><strong>39.3</strong></td>
    <td><strong>37.7</strong></td>
    <td><strong>27.0</strong></td>
    <td><strong>69.2</strong></td>
    <td><strong>87.5</strong></td>
    <td>50.3</td>
    <td style="border-left:1px solid; border-right:3px double;"><strong>47.2</strong></td>
    <td><strong>31.7</strong></td>
    <td>16.9</td>
    <td>40.6</td>
    <td>4.2</td>
    <td>100</td>
    <td><strong>99.6</strong></td>
  </tr>
  <!-- LLaMA 3.1 Rows -->
  <tr>
    <td><strong>LLaMA 3.1</strong></td>
    <td>InfLLM (4k+4k)</td>
    <td><strong>41.4</strong></td>
    <td>40.7</td>
    <td>29</td>
    <td>69</td>
    <td>97</td>
    <td><strong>64.2</strong></td>
    <td style="border-left:1px solid; border-right:3px double;">51.1</td>
    <td>22.6</td>
    <td>33.7</td>
    <td>46.7</td>
    <td>81</td>
    <td>100</td>
    <td>100</td>
  </tr>
  <tr>
    <td></td>
    <td>EM-LLM<sub>SM</sub></td>
    <td>41.2</td>
    <td><strong>41.3</strong></td>
    <td><strong>29.2</strong></td>
    <td><strong>69.1</strong></td>
    <td><strong>98.5</strong></td>
    <td>64.1</td>
    <td style="border-left:1px solid; border-right:3px double;"><strong>51.3</strong></td>
    <td>22.6</td>
    <td><strong>34</strong></td>
    <td><strong>47.6</strong></td>
    <td><strong>90.2</strong></td>
    <td>100</td>
    <td>100</td>
  </tr>
  <!-- Subheader: 4B parameter models -->
  <tr>
    <td colspan="15" align="center"><strong>4B parameter models</strong></td>
  </tr>
  <!-- Phi 3 Rows -->
  <tr>
    <td><strong>Phi 3</strong></td>
    <td>InfLLM (1k+3k)</td>
    <td>28.4</td>
    <td>24.9</td>
    <td>25.6</td>
    <td>52.9</td>
    <td>7.5</td>
    <td>57</td>
    <td style="border-left:1px solid; border-right:3px double;">34.5</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>EM-LLM<sub>S</sub></td>
    <td><strong>29.2</strong></td>
    <td><strong>27.1</strong></td>
    <td><strong>25.9</strong></td>
    <td><strong>53.5</strong></td>
    <td><strong>10</strong></td>
    <td>57</td>
    <td style="border-left:1px solid; border-right:3px double;"><strong>35.4</strong></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <!-- Phi 3.5 Rows -->
  <tr>
    <td><strong>Phi 3.5</strong></td>
    <td>InfLLM (1k+3k)</td>
    <td>31.7</td>
    <td>28.5</td>
    <td>23.9</td>
    <td><strong>56.3</strong></td>
    <td>11.5</td>
    <td><strong>40.3</strong></td>
    <td style="border-left:1px solid; border-right:3px double;">34.2</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>EM-LLM<sub>S</sub></td>
    <td><strong>31.8</strong></td>
    <td><strong>31.9</strong></td>
    <td><strong>24.5</strong></td>
    <td>55.5</td>
    <td><strong>13</strong></td>
    <td>39.5</td>
    <td style="border-left:1px solid; border-right:3px double;"><strong>34.9</strong></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

</center>

**Table 2:** EM-LLM performance on LongBench (grouped tasks) and $\infty$-Bench compared to our baseline InfLLM. **S**: surprise threshold, **SM**: surprise threshold and refinement with modularity, **S+C**: surprise threshold and contiguity buffer, **SM+C**: surprise, refinement and contiguity buffer. Each row indicates the number of local + retrieved tokens (e.g., 4k+2k) used for both InfLLM and EM-LLM.