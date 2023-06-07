# import pymol
import __main__
import os
import subprocess

import numpy as np
import pandas as pd

# from pymol import cmd
from task.base import BaseTool


############################
# Black Box Tools
############################

class Absolut(BaseTool):
    def __init__(self,
                 config):
        BaseTool.__init__(self)
        '''
        config: dictionary of parameters for BO
            antigen: PDB ID of antigen
            path: path to Absolut installation
            process: Number of CPU processes
            expid: experiment ID
        '''
        for key in ['antigen', 'path', 'process']:
            assert key in config, f"\"{key}\" is not defined in config"
        self.config = config
        assert self.config['startTask'] >= 0 and (self.config['startTask'] + self.config['process'] < os.cpu_count()), \
            f"{self.config['startTask']} is not a valid cpu"

    def Energy(self, x):
        '''
        x: categorical vector (num_Seq x Length)
        '''
        x = x.astype('int32')
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # randomise TempCDR3 name in case 2 or more experiments are run in parallel on the same server
        # rand_num = np.random.randint(low=0, high=1000)

        # Change working directory
        current_dir = os.getcwd()
        os.chdir(f"{self.config['path']}")

        sequences = []
        with open(f"TempCDR3_{self.config['antigen']}.txt", "w") as f:
            for i, seq in enumerate(x):
                seq2char = ''.join(self.idx_to_AA[aa] for aa in seq)
                line = f"{i + 1}\t{seq2char}\n"
                f.write(line)
                sequences.append(seq2char)

        _ = subprocess.run(
            ['taskset', '-c', f"{self.config['startTask']}-{self.config['startTask'] + self.config['process']}",
             "./src/bin/Absolut", 'repertoire', self.config['antigen'], f"TempCDR3_{self.config['antigen']}.txt",
             str(self.config['process'])], capture_output=True, text=False)

        data = pd.read_csv(os.path.join(self.config['path'],
                                        f"{self.config['antigen']}FinalBindings_Process_1_Of_1.txt"),
                           sep='\t', skiprows=1)

        # Add an extra column to ensure that ordering will be ok after groupby operation
        data['sequence_idx'] = data.apply(lambda row: int(row.ID_slide_Variant.split("_")[0]), axis=1)
        energy = data.groupby(by=['sequence_idx']).min(['Energy'])
        min_energy = energy['Energy'].values

        # Remove all created files and change the working directory to what it was
        for i in range(self.config['process']):
            os.remove(f"TempBindingsFor{self.config['antigen']}_t{i}_Part1_of_1.txt")
        os.remove(f"TempCDR3_{self.config['antigen']}.txt")

        os.remove(f"{self.config['antigen']}FinalBindings_Process_1_Of_1.txt")
        os.chdir(current_dir)
        return min_energy, sequences


class AbsolutVisualisation:
    def __init__(self, config):
        self.config = config

        assert 'AbsolutPath' in config, "\"AbsolutPath\" is not defined in config"

    def visualise(self, antigen, CDR3):
        """
        antigen: PDB ID of antigen
        CDR3: String that specifies the CDR3
        """
        current_dir = os.getcwd()
        os.chdir(f"{self.config['AbsolutPath']}")
        output = subprocess.run(["./src/bin/Absolut", "info_fileNames", antigen], capture_output=True, text=True)
        if output.returncode != 0:
            raise Exception("Specified antigen is not in the library. Try another antigen or check the antigen ID.")

        # Check if precalculated structures are downloaded
        antigen = output.stdout.split(" ")[6]
        precalculated_structure_filename = output.stdout.split(" ")[13].split(".txt")[0] + ".txt"
        if not os.path.exists(os.path.join(self.config['AbsolutPath'], precalculated_structure_filename)):
            print('Downloading Precalculated antigen structure...')
            subprocess.run(
                ["wget", "http://philippe-robert.com/Absolut/Structures/" + precalculated_structure_filename],
                capture_output=True, text=True)
            # After downloading the files have to run the command below to generate some additional files. This can
            # Take some time
            output = subprocess.run(["./src/bin/Absolut", "singleBinding", antigen, CDR3], capture_output=True,
                                    text=False)

        # Get the antibody-antigen complex of the CDR3 sequence to an antigen
        print('Predicting CDR3 docking...')
        with open(f"TempCDR3_{antigen}.txt", "w") as f:
            line = f"{0 + 1}\t{CDR3}\n"
            f.write(line)
        output = subprocess.run(["./src/bin/Absolut", "repertoire", antigen, f"TempCDR3_{antigen}.txt"],
                                capture_output=True, text=False)

        # It is necessary to set this variable to enable GUI
        os.environ["DISPLAY"] = "127.0.0.1:10.0"
        output = subprocess.run(
            ["./src/bin/Absolut", "visualize", antigen, f"{antigen}FinalBindings_Process_1_Of_1.txt"])

        # Remove generated files
        os.remove(f"TempCDR3_{antigen}.txt")
        os.remove(f"TempBindingsFor{antigen}_t{0}_Part1_of_1.txt")
        os.remove(f"{antigen}FinalBindings_Process_1_Of_1.txt")
        os.chdir(current_dir)


class PyMolVisualisation:
    def __init__(self):
        pass

    def visualise(self, antigen, video_length=2):
        """
        VLC player required to view output video on windows
        """
        num_frames = video_length * 30
        # It is necessary to set this variable to enable GUI
        os.environ["DISPLAY"] = "127.0.0.1:10.0"

        # Tell PyMOL to launch quiet (-q), fullscreen (-e), without internal GUI (-i)
        __main__.pymol_argv = ['pymol', '-qei']
        pymol.finish_launching()

        # For now download the antigen from the PDB. Later, once getAntibody and dockAntibody are implemented, load the
        # complex instead with cmd.load("{}.pdb".format(antigen))
        cmd.do(f"fetch {antigen}")

        # Create a short movie of the complex rotating using PyMol commands
        cmd.do(f"""
        remove solvent
        bg_color white
        extract ligands, het
        spectrum count, green_yellow_red
        color blue, ligands
        set cartoon_fancy_helices, 1
        
        mset 1 x{num_frames}
        util.mroll 1, {num_frames}, 1
        set ray_trace_frames, 1
        set cache_frames, 0
        movie.produce {antigen}.mpg, quality=90
        """)

        # additional things we can set. This should go before mset 1 x{}
        # cmd.do("""
        # set ray_trace_mode, 1
        # set ray_shadow, 1
        # set light_count, 2
        # set light, [0, 0, -100]
        # set ambient, 0
        # set direct, 0.7
        # set specular, 1
        # set shininess, 5
        # set specular_intensity, 0.3
        # set reflect, 0.2
        # set reflect_power, 1
        # set depth_cue, 1
        # set fog_start, 0.45
        # set antialias, 3
        # """)

        cmd.quit()


class Manual(BaseTool):
    def __init__(self, config):
        BaseTool.__init__(self)
        '''
        config: dictionary of parameters for BO
            antigen: PDB ID of antigen
            process: Number of CPU processes
            expid: experiment ID
        '''
        for key in ['antigen']:
            assert key in config, f"\"{key}\" is not defined in config"
        self.config = config
        self.antigen = self.config["antigen"]

    def Energy(self, x):
        '''
        x: categorical vector (num_Seq x Length)
        '''
        x = x.astype('int32')
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        print("Suggested sequences:")
        sequences = []
        for i, seq in enumerate(x):
            seq2char = ''.join(self.idx_to_AA[aa] for aa in seq)
            sequences.append(seq2char)
            print(seq2char)

        energies = []
        for i in range(len(sequences)):
            energy1 = float(input(f"[{self.antigen}] Write energy for {sequences[i]}:"))
            energy2 = float(input(f"[{self.antigen}] Confirm energy for {sequences[i]}:"))
            while energy1 != energy2:
                print("Mismatch, pleaser enter energies again")
                energy1 = float(input(f"[{self.antigen}] Write energy for {sequences[i]}:"))
                energy2 = float(input(f"[{self.antigen}] Confirm energy for {sequences[i]}:"))
            energies.append(energy1)

        return np.array(energies), sequences


###########################################
# Docking and Visualisation Tool
###########################################

class Visualisation:
    def __init__(self,
                 antigen,
                 docktool='ClusPro',
                 vistool='PyMol'):
        '''
        antigen: PDB ID of an antigen
        '''
        self.antigen = antigen
        self.docktool = docktool
        self.vistool = vistool

    def replace_CDR3(self, antiseq, x, y):
        '''

        :antiseq: Antibody as a sequence of Amino Acids
        :x: CDR3 of an antibody
        :y: new CDR3
        :return:
        '''
        raise NotImplementedError

    def load_fasta(self, y):
        '''
        y: pdb id
        return:
            FASTA Sequence
        '''
        raise NotImplementedError

    def dockAntibody(self, x):
        '''
        x: pdb file name of antibody
        '''
        raise NotImplementedError

    def visualise(self, x, y):
        '''
        x: CDR3  Sequence
        y: Antibody ID
        visualise binding
        '''
        z = self.getAntibody(x, y)
        w = self.dockAntibody(z)
        self.vistool.visualise(w)
