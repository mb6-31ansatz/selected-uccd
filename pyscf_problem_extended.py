# Copyright 2026 Michelangelo Bovoloni, Piero Macchi, Antonino Famulari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Statevector

import pandas as pd
import math
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations

from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType

from qiskit.quantum_info import Pauli

import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

# big check
#from qiskit.opflow import I, Z, X, Y  #, PauliOp
from qiskit.circuit import ParameterVector

from qiskit.quantum_info import Pauli

# VQE PART
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
#from qiskit.primitives import StatevectorEstimator as Estimator #Estimator
from qiskit_aer.primitives import EstimatorV2 as  Estimator #Estimato
#from qiskit_nature.second_q.algorithms import GroundStateEigensolver, QEOM, EvaluationRule
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from qiskit_nature.second_q.circuit.library import HartreeFock, UCC, UCCSD
import time

from qiskit.quantum_info.operators import Operator, Pauli

from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations

cc = 'Explanation'

   
#################################################################################################################
################################################# MOLECULE CLASS
##############################################################################################################
from enum import Enum

class InitialGuess(Enum):
    """Initial Guess Enum"""

    MINAO = "minao"
    HCORE = "1e"
    ONE_E = "1e"
    ATOM = "atom"



class Molecule_simul_new_mapping():
    
    def __init__(self, molecule_xyz, basis_AO, charge, spin, method_type ):
        
        self.molecule_xyz = molecule_xyz
        self.basis_AO = basis_AO
        self.charge = charge
        self.spin = spin
        self.method_type = method_type
        
        
    #####################################################
    ###############  TOOLBOX  ###########################
    #####################################################
    
    def excitation_str_readable(self):
        """This function is able to convert the ucc_operator.excitation_list into a more Readable excitation list with alpha and beta spin.
        From qiskit, in the ucc_operator.excitation_list the alpha spin are the 0 to M-1 and beta spin from M to 2*M-1 value. 
        Where M is the number of Spatial Orbital.
        
        RETURN: Dictionary with Excitation Position in the Hamiltonian Matrix and Excitation string readable format."""
        
        # Dictionary that gives me all value_dict
        configurations_value_dict = {}
                
        ################################################
        for excitation_i in self.excitation_lst: #[start: end]:
            
            # Here the Annhilation
            electron_removal_tuple = excitation_i[0]
            
            # Here the Cretion eletron
            electron_addition_tuple = excitation_i[1]
            
            ######################################
            new_data = self.hf_config .copy()
            #print(new_data)
            
            for i in electron_removal_tuple:
                ##############################################
                new_data.remove(self.spins_label_dict[i])
                
                
            for j in electron_addition_tuple:
                ##############################################
                new_data.append(self.spins_label_dict[j])
                        
            #############################################
            #print(self.value_dict_spin_orb)
            
            value_string = sum([self.value_dict_spin_orb[k] for k in new_data ])
            #print('BANG2:\t', value_string, '\n')
            
            
            configurations_value_dict[value_string] = new_data
                        
        return configurations_value_dict
    
    def show_excitation_pauli_strings(self):
        """ Show the Exciatation Pauli Strings used to build the Excitation Operator in the UCC Ansatz
        self.excitation_operators = self.ucc_operator.excitation_ops()   
        # B. Creation of Excitation Pauli Strings SAME as the UCC 
        # in the JW MAPPER
        self.excitation_pauli_strings = [ self.mapper.map(exc_op) for exc_op in self.excitation_operators]
        
        # C. Create a NEW MAPPER that uses the Zsymmetries of the system, starting from the JW Mapper
        self.mapper_tap = self.problem.get_tapered_mapper( self.mapper )
        # D. Repeat step B and hence create the Excitation Pauli Strings SAME as the UCC 
        # in the REDUCED MAPPER
        self.excitation_pauli_strings_map = [ self.mapper_tap.map(exc_op) for exc_op in self.excitation_operators]
        """
        print('JW Excitations Strings')
        for i in range(len( self.excitation_pauli_strings )):
            print(f'Excitation {i} and Pauli strings in JW is:\n', 
                  self.excitation_pauli_strings[i], '\n' )
            
        print('TAPERING Excitations Strings')
        for j in range(len( self.excitation_pauli_strings_map )):
            print(f'Excitation {j} and Pauli strings in TAPERING is:\n', 
                  self.excitation_pauli_strings_map[j], '\n' ) 

        
    def creation_disperse_array(self, param_label):
        
        """self.df1_in_dict = {}
        
        for i in gg.df1.index:
            self.df1_in_dict[ self.df1['0_x'].to_dict()[i] ] = self.df1[ param_label ].to_dict()[i]"""
        
        self.df1_in_dict = self.df1[ param_label ].to_dict()
            
        full_disperse_array = [0] * self.ucc_operator.num_parameters
    
        #print(self.df1_in_dict[i])
        for i in self.df1_in_dict.keys():
            full_disperse_array[i] = self.df1_in_dict[i]
    
        #print(full_disperse_array)
        return full_disperse_array
    


    #####################################################
    ############### END TOOLBOX  ########################
    #####################################################
    
    
    def problem_created(self, frozen_core = 'Y', active_space = {'active': 'N' , 'num_active_orbitals':10} ):
        """Creation of all necessary tools for future analysis:
        1. HF Configuration. Called as an Attribute:  hf_config
        
        2. LABEL to all SPIN-ORBITALS in a DICT. Called as an Attribute: spin_orbitals_lst
        
        3. Dictionary that keep track of the correspondance between the spin-orbitals i and its corresponding position value in the JW Matrix.
        This postion is given as 2**N where N = {0, Num_qubits - 1} starting from the lower energy level,
        then climbing up, FIRSTLY considering only THE ALPHA - SPIN then the BETAs. Called as an Attribute: value_dict_spin_orb
        
        4. Dictionary in which I assign a POSTION value, in an given SPATIAL ORBITAL LADDER, ranging from 0 to 2*M-1 
        (where M is the number of spatial orbitals) CORRESPONDING TO SPIN Oribital CONFIGURATION. This an helpful dictionary 
        used for when there will be the Excitations in a list. Called as an Attribute: spins_label_dict
        
        5. Postion Value in the JW Matrix for the HF Configuration. Called as an Attribute:  hf_config_value
        """
            
        """driver = PySCFDriver(
        atom= self.molecule_xyz, 
        basis= self.basis_AO,   
        charge= self.charge,
        spin= self.spin,
        unit=DistanceUnit.ANGSTROM, 
        method = self.method_type, 
        xc_functional = 'B3LYP',    #############    CAMB3LYP, PBE0, PBE, TPSS, LDA
        #init_guess = InitialGuess.ATOM,
        max_cycle= 10000)"""

        #self._mol.symmetry = False
        
        #print('\n driver._build_molecule --_calc.mol.symmetry\n', driver._build_molecule)
        #print('\n!!!!!!')

        #import importlib
        #importlib.reload(pyscf_problem)

        from . import PySCFDriver_mod_vers
        import importlib
        importlib.reload(PySCFDriver_mod_vers)

        driver = PySCFDriver_mod_vers.PySCFDriver(
        atom= self.molecule_xyz, 
        basis= self.basis_AO,   
        charge= self.charge,
        spin= self.spin,
        unit=DistanceUnit.ANGSTROM, 
        method = self.method_type, 
        xc_functional = 'B3LYP',    #############    CAMB3LYP, PBE0, PBE, TPSS, LDA
        #init_guess = InitialGuess.ATOM,
        max_cycle= 10000)

        #print(' InitialGuess.ATOM',  InitialGuess.ATOM)
        
        # Problem Created
        self.save_driver = driver
        self.problem = driver.run()
                    
            
        self.repulsion = self.problem.hamiltonian.constants['nuclear_repulsion_energy']

        
        self.frozen_core ='N'
        
        # Removal FrozenCore from Atoms:  
        if frozen_core == 'Y':
            fc_transformer = FreezeCoreTransformer()

            self.problem = fc_transformer.transform(self.problem)
            
            print('Frozen Core informations:',self.problem.hamiltonian.constants)
            
            # Saving parts
            self.core = self.problem.hamiltonian.constants['FreezeCoreTransformer']
            # Saving parts
            self.frozen_core ='Y'
        
            
            
        if active_space['active'] == 'Y': #(self.problem.num_particles
            self.problem = ActiveSpaceTransformer(self.problem.num_particles , active_space['num_active_orbitals'] ).transform(self.problem)
            


        ############################jjjjj
        # Number 1 HF Configuration
        self.hf_config = sorted([f'{i}A' for i in range(1, self.problem.num_particles[0] +1) 
                         ]+ [f'{i}B' for i in range(1, self.problem.num_particles[1] +1) ])
        
        # Number 2 All spin- orbitals given the basis
        self.spin_orbitals_lst = [f'{i}{j}' for j in ['A', 'B'] for i in range(1,self.problem.num_spatial_orbitals+1)]
        
        print(self.spin_orbitals_lst)
        print('\n')

        # Number 3 Spin-orbitals and Position value
        self.value_dict_spin_orb = {self.spin_orbitals_lst[i]:2**i for i in range(len(self.spin_orbitals_lst ))}  
        
        # Number 4 Excitation in Spatial Orbital Ladder and spin-orbitals 
        self.spins_label_dict = {i: f'{i+1}A' for i in range( self.problem.num_spatial_orbitals )} 
        self.spins_label_dict.update({i: f'{i-self.problem.num_spatial_orbitals+1 }B' 
                                            for i in range( self.problem.num_spatial_orbitals, self.problem.num_spatial_orbitals*2 )})
        
        # Number 5 Position value for HF Configuration
        self.hf_config_value = sum([ self.value_dict_spin_orb[i] for i in self.hf_config ])
        
        
        # NUMBER OF OCCUPIED ORBITALS
        self.occupied_orbitals = math.ceil(sum(self.problem.num_particles)/2)
        
        # NUMBER OF UNOCCUPIED ORBITALS
        self.unoccupied_orbitals = self.problem.num_spatial_orbitals - self.occupied_orbitals
        
        
        self.recap_problem_created = f"""Other important parts for the problem created now.
        
        A. Number of spin orbitals is: {self.problem.num_spin_orbitals}
        B. The number of JW configuration is: 2^{self.problem.num_spin_orbitals} =  {2**self.problem.num_spin_orbitals}
        C. Hence the JW matrix is a: {self.problem.num_spin_orbitals} x {self.problem.num_spin_orbitals}
        D. Occupied orbitals: {self.occupied_orbitals}
        E. Unoccupied orbitals: {self.unoccupied_orbitals}
        """
        
        print('\n\n',self.recap_problem_created, '\n'  )



        return self.problem
        

    def excitations_Dataframe(self, excitation_str = 'sd'):
        """Here,creation of the EXCITATIONS DATAFRAME
        1. column, using the function excitation_str_readable(), I get a dictionary with the Location of the 
        given excitation element ij in the Hamiltonian Matrix mapped by the JW Transformation as keys.
        2. column from the same dictionary the values are the Excitation written in a chemist friendly string.
        
        3. column represents the electrons removed from n given SPIN-ORBITALS by the process of excitation
        4. column represents the electrons relocated into n given SPIN-ORBITALS by the process of excitation
        The combination of 3-4 column represents the same reality of column 2. 
        In this case, however, the First number ALWAYS Refers to the ALPHAs spin*-orbital and the Second to the 
        BETAs. Their numeration goes from 0 to M-1, for ALPHAs, and from M to 2*M-1, for BETAs. Where, M is the 
        number of SPATIAL ORBITALS given by the Basis Set choosen.
        
        5. column that keep track if the excitation i should be kept or discarded due to Zsymmetries
        6. column in which for each excitation i, it is reported the ij matrix element of the Linar Combination 
        of Pauli matrices, used as base to create the EXCITATION OPERATOR for that Specific ANSATZ (UCC).
        This element ij from the LC of Pauli matrices to form a single excitation operator, it will either be:
        +1j or -1j, hence giving the sign of the excitation i.
        This position ij is given by i = row of the Hartree-Fock element, j = column of the Excitation i wanted.
    
        RETURN a Reduced Dataframe, having the as Index the i-th excitation appearing in the UCC Ansatz,
        using a JW Mapper. Then the position of the Excitiation element matrix in the JW matrix, 
        the configuration associated to that Excitation element both in a chemist and second quantization view. 
        Then the sign associated to the rotation associated to that particular excitation. 
        Finally in the last column, this excitation will have a value == keep if it is an important excitation
        or it can be discarded due to Zsymmetries, given by the tapering method"""
        
        
        # 3 and 4 column Excitation List
        ###################
        
        excitation_str_dict = {'s': 1, 'd':2, 't':3, 'q':4, '5':5, '6':6 }
        

        
        excitation_lst = []
        for exc_i in list(excitation_str ):
            try:
                exc_input = excitation_str_dict[exc_i]
            
                excitation_lst += generate_fermionic_excitations(num_excitations= exc_input , 
                                       num_spatial_orbitals= self.problem.num_spatial_orbitals, 
                                       num_particles= self.problem.num_particles )
                
            except:
                excitation_lst += generate_fermionic_excitations(num_excitations= exc_i , 
                                       num_spatial_orbitals= self.problem.num_spatial_orbitals, 
                                       num_particles= self.problem.num_particles )

            
        self.excitation_lst = excitation_lst

        temp_df2 = pd.DataFrame( excitation_lst  )
        self.temp_df2 = temp_df2
        ###################
        
        
        
        # 1 and 2 column
        ##################
        exctitation_dict = self.excitation_str_readable()        
        exctitation_list = [  ( i, exctitation_dict[i] ) for i in list( exctitation_dict.keys() ) ] 
        
        temp_df1 = pd.DataFrame( exctitation_list )
        self.temp_df1 = temp_df1
        ###################
        
        

        
        
        # 5 Column keep ALL -- REDUCTION LATER
        ###################
        
        ###########  CREATION DATAFRAME  ###########
        
        temp_df_fin = pd.merge(temp_df1, temp_df2, left_index=True, right_index=True)
        
        self.df_exc = temp_df_fin
    
        
        
        
    def problem_mapped_JW(self, excitations_type):
        """Function to Map the problem from Fermionic representation to Qubit representation.
        The mapping used for this kind of transformation is the Jordan-Wigner transformation.
        The Ansatz used is the Unitary Couple Cluster (UCC) with a chosen number of wanted excitations
        given thanks to the keyword excitations_type."""
        # Mapper Used
        self.mapper  = JordanWignerMapper()
        #print(self.mapper)
            
        t1 = time.time()
        # Hamiltonian Problem mapped in JW mapping
        self.JW_problem_mapped = self.mapper.map(self.problem.hamiltonian.second_q_op() )
        self.test = JordanWignerMapper().map(self.problem.hamiltonian.second_q_op() )
        
        #print(self.JW_problem_mapped)
        delta_t = time.time() - t1
        print('Time for JW Mapping: ',delta_t, '\n')
        
        # Num of qubits
        self.num_qubits = self.JW_problem_mapped.num_qubits
        
        # excitation type
        self.excitations_type =  excitations_type
        
        # UCC OPERATORS
        ucc_ = UCC(
            self.problem.num_spatial_orbitals,
            self.problem.num_particles, 
            self.excitations_type ,
            self.mapper,
            #generalized=True,
            
            initial_state=HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper),)
        
        self.ucc_operator = ucc_
        
        
    def problem_reduction(self, reduction_in_place = 'active'):
        """Here, using the Zsymmetries of the problem (Hamiltonian Problem) it is possible to reduce 
        the number of excitations of interest. Then thanks to a set of operation it is possible to obtain 
        a DataFrame containing all the information of interest for a GIVEN Excitation.
        Here a list of operation to get each column, in order:
        1. column, using the function excitation_str_readable(), I get a dictionary with the Location of the 
        given excitation element ij in the Hamiltonian Matrix mapped by the JW Transformation as keys.
        2. column from the same dictionary the values are the Excitation written in a chemist friendly string.
        
        3. column represents the electrons removed from n given SPIN-ORBITALS by the process of excitation
        4. column represents the electrons relocated into n given SPIN-ORBITALS by the process of excitation
        The combination of 3-4 column represents the same reality of column 2. 
        In this case, however, the First number ALWAYS Refers to the ALPHAs spin*-orbital and the Second to the 
        BETAs. Their numeration goes from 0 to M-1, for ALPHAs, and from M to 2*M-1, for BETAs. Where, M is the 
        number of SPATIAL ORBITALS given by the Basis Set choosen.
        
        5. column that keep track if the excitation i should be kept or discarded due to Zsymmetries
        6. column in which for each excitation i, it is reported the ij matrix element of the Linar Combination 
        of Pauli matrices, used as base to create the EXCITATION OPERATOR for that Specific ANSATZ (UCC).
        This element ij from the LC of Pauli matrices to form a single excitation operator, it will either be:
        +1j or -1j, hence giving the sign of the excitation i.
        This position ij is given by i = row of the Hartree-Fock element, j = column of the Excitation i wanted.
    
        RETURN a Reduced Dataframe, having the as Index the i-th excitation appearing in the UCC Ansatz,
        using a JW Mapper. Then the position of the Excitiation element matrix in the JW matrix, 
        the configuration associated to that Excitation element both in a chemist and second quantization view. 
        Then the sign associated to the rotation associated to that particular excitation. 
        Finally in the last column, this excitation will have a value == keep if it is an important excitation
        or it can be discarded due to Zsymmetries, given by the tapering method"""
        
        
        # 1 and 2 column
        ##################
        exctitation_dict = self.excitation_str_readable()        
        exctitation_list = [  ( i, exctitation_dict[i] ) for i in list( exctitation_dict.keys() ) ] 
        
        dd = pd.DataFrame( exctitation_list )
        ###################
        
        
        # 3 and 4 column Excitation List
        ###################
        dd2 = pd.DataFrame( self.ucc_operator.excitation_list  )
        ###################
        
        
        # 5 Column keep or discard excitation
        ###################
        # A. Extract the excitation operators corresponding to the UCC operator
        self.excitation_operators = self.ucc_operator.excitation_ops()   
        
        # B. Creation of Excitation Pauli Strings SAME as the UCC 
        # in the JW MAPPER
        self.excitation_pauli_strings = [ self.mapper.map(exc_op) for exc_op in self.excitation_operators]
        
        t1 = time.time()
        # C. Create a NEW MAPPER that uses the Zsymmetries of the system, starting from the JW Mapper
        self.mapper_tap = self.problem.get_tapered_mapper( self.mapper )

        #
        delta_t = time.time() - t1
        print('Time for Tapering Symmerties: ',delta_t, '\n')
        
        # D. Repeat step B and hence create the Excitation Pauli Strings SAME as the UCC 
        # in the REDUCED MAPPER
        self.excitation_pauli_strings_map = [ self.mapper_tap.map(exc_op) for exc_op in self.excitation_operators]
        
        # E. Create a List of that recap if an excitation i is KEPT or DISCARDED based on the ZSymmetries found
        # using the NEW MAPPER
        exctitation_list_presence = []
        for i in self.excitation_pauli_strings_map:
            if i !=None:
                exctitation_list_presence.append('keep')
            else:
                exctitation_list_presence.append('remove')
        
        # F. Later add *exctitation_list_presence* in the Final DataFrame 
        ###################
        
        
        # 6 column - SIGN of the Excitation i
        ###################
        # A. Call the Tensor_matrix_toolbox class with the number of qubits used in the Jordan Wigner mapping
        #ff = Tensor_matrix_toolbox( self.num_qubits )
        
        # B. Start a loop over all excitation i, where their positions jj in the Hamiltonian matrix mapped
        # by JW transformation are stored in the DataFrame dd.
        # save these signs into the sign_list
        sign_list = []
        row = self.hf_config_value
        for exc_op_i in range(len(self.excitation_pauli_strings)):
            
            col = dd.iloc[ exc_op_i , 0]
            data = 0#ff.energy_or_exc_element_ij( row, col, self.excitation_pauli_strings[exc_op_i] )
            
            sign_list.append(data)
        
        # C. Later add *sign_list* in the Final DataFrame 
        ###################
        

        
        ###########  CREATION DATAFRAME  ###########
        
        df1 = pd.merge(dd, dd2, left_index=True, right_index=True)
        
        # F. Later add *exctitation_list_presence* in the DataFrame 
        df1['keep_excitation'] = exctitation_list_presence
        
        # H. ADDITION OF Both JW and Tapering Pauli Strings
        #df1['JW strings'] = self.excitation_pauli_strings
        #df1['Tapering strings'] = self.excitation_pauli_strings_map
        
        # C. Later add *sign_list* in the DataFrame
        #df1['sign exc t[i]'] = np.array(sign_list) 
        df1['sign exc t[i]'] = np.array(sign_list) * -1j
        

        if reduction_in_place == 'inactive':
            print('Elimination of Useless column saved into: df2')
            self.df2 = df1.copy()

        print('Elimination of Useless column Active in df1')
        df1 = df1.drop( df1[df1['keep_excitation'] == 'remove'].index )
        
        self.df1 = df1 
        #return  self.df1
    

    def problem_parameters(self):
        """PART1: The first Snippet of code to to extract the energies ij and jj. In this case these 2 elements constitute a 
         2 by 2 Hermitian matrix, where the ii element is the HF energy, the ij == ji element represents the 
         interaction between the HF configuration and the Excitation j configuration. Finally the element jj is
         the energy of the Excitation jj.
        
         PART 2: The second Snippet of code is designed to extract the Rotation Parameter t to DIAGONALIZE a 2 by 2 matrix
         linked to the specific excitation j interacting ALONE with the HF configuration. 
         This Diagonalization of an initial Undiagonalized Matrix is done through a 2 by 2 UNITARY MATRIX built 
         using the PARAMETER **t**. 
         IT IS Possible, furthermore, to find this variable **t** by finding directly the first or second 
         Eigenvector element (and takning the arccos and arcsin respectively) of an Hermitian Matrix, 
         obtained by using a BLACK BOX DIAGONALIZATION PROCESS.
         
         PART 3: This third Snippet of code is a necessary piece in the overall code, since it is able to determine
         the TRUE SIGN of the Parameter found. In the previous step, ONLY THE REDUCED HERMITIAN MATRIX is used 
         to find the WANTED EIGENVECTOR, through a Black box. This action, however, it is not the COMPLETELY 
         EQUIVALENT to the previous one, because the HERMITIAN MATRIX ALONE DO NOT HAVE the PHASE of the ROTATION 
         employed in the ALGORITHM, that I want SIMULATE in a LOWER DIMENSION. 
         Therefore, this MISMATCH in Phase, lead to unproper results. 
         IN ORDER TO REMOVE This ISSUE, it is necessary to SIMULATE in a LOWER DIMENSION This ROTATION, employing
         an EQUIVALENT UNITARY MATRIX (in low dimension). 
         To have this EQUIVALENT UNITARY MATRIX I need 2 element. The parameter **t**, which it will be Evaluated
         and the SIGN I FIND from the Linear Combination of Pauli strings used, as the main component, to DETERMINE
         the ROTATION OF THE EXCITATION j.
             NB Thanks to part 3 I am able to get the correct value of the Parameter, but not its Phase. 
             Hence, I get the Modulus but not the Phase.
        """

        
        
        # PART 1 EXTRACTION IMPORTANT ENERGIES ij and jj
        ###################
        # 1 column - SIGN of the Excitation i
        ###################
        # A. Call the Tensor_matrix_toolbox class with the number of qubits used in the Jordan Wigner mapping
        ff = Tensor_matrix_toolbox( self.num_qubits )
        
        # B. Start a loop over all excitation i, where their positions jj in the Hamiltonian matrix mapped
        # by JW transformation are stored in the DataFrame dd.
        # save these signs into the sign_list
        
        """energy_ij_lst = []
        energy_jj_lst = []
        
        row = self.hf_config_value
        for exc_i in range(len ( self.df1['0_x'] ) ):
            
            if len(self.df1['1_y'].iloc[exc_i]) == 2:  
            
                col = self.df1.iloc[ exc_i , 0]
                data_ij = ff.energy_or_exc_element_ij( row, col, self.JW_problem_mapped )
                energy_ij_lst.append(data_ij.real)
            
                data_jj = ff.energy_or_exc_element_ij( col, col, self.JW_problem_mapped )
                energy_jj_lst.append(data_jj.real)
                
            else:
                energy_ij_lst.append(0)
                energy_jj_lst.append(0)


        self.df1['energy ij'] = energy_ij_lst
        self.df1['energy jj'] = energy_jj_lst"""

    
        
        
        # PART 2 ESTRAPOLATION IMPORTANT PARAMETERS for Excitation j-th
        ###################
        row = self.hf_config_value
        self.hf_config_energy_ii = ff.energy_or_exc_element_ij( row, row, self.JW_problem_mapped ).real
        
        start_matrix = np.zeros([2,2])
        start_matrix[0,0] = self.hf_config_energy_ii
        
        """param_lst = []
        param_lst2 = []
        for exc_i in self.df1.index: 
            #print(exc_i)
            pos_ij = self.df1.loc[ exc_i , 'energy ij']
            pos_jj = self.df1.loc[ exc_i , 'energy jj']
            
            start_matrix[0,1] = pos_ij
            start_matrix[1,0] = pos_ij
            start_matrix[1,1] = pos_jj
            
            eigenvalues, eigenvectors = np.linalg.eig( start_matrix )
            data = eigenvectors[:, 0][-1]
            
            param_lst.append( np.arcsin(data) )
            #param_lst2.append( data )
        
        
        self.df1['t[i] arcsin'] = abs(np.array(param_lst))"""
        #self.df1['t[i] pure'] = param_lst2
        #self.df1['t[i] pure sign'] = abs(np.array(param_lst2)) * np.array(self.df1['sign exc t[i]'] ).real


        # PART 3 - Correct Determination of the Parameter t sign.
        ###################
        """value_dict_param_true = {}
        
        for index in self.df1.index:
            # 3 parameters for the New Matrix for the ESTIMATION
            param = [self.hf_config_energy_ii, 
                     self.df1.loc[index, 'energy ij'], self.df1.loc[index, 'energy jj']]
            
            #  Take the sign -----------------------------------------------------------------------------
            sign = (self.df1.loc[index, 'sign exc t[i]'] * -1j).real
            
            # each paramaters getted 
            t = self.df1.loc[index, 't[i] arcsin']

            # each parameters is tested to see WHICH IS THE RIGHT SIGN OF THE PARAMETER
            val = dot_matrix(param, sign, t)
            
            # Put the value in the value_dict_param_true for the PARAMETERS
            # created list of Parameters for the ESTIMATION 
            value_dict_param_true[index] = val
        
        self.df1['t[i] arcsin tested'] = list(value_dict_param_true.values() )
        
        self.df1['sign param'] = self.df1['sign exc t[i]'] * -1j"""
        #------------------------------------------------------------------------------
        ###################
        
        self.estimator = Estimator()
        
        t1 = time.time()
        self.hamiltonian_tapering = self.mapper_tap.map( self.problem.hamiltonian.second_q_op() )
        self.hamiltonian_JW = self.mapper.map( self.problem.hamiltonian.second_q_op() )
        delta_t = time.time() - t1
        print('time for mapper NON TAPERING:', delta_t)

    
    
    def problem_resolution_JW(self, test='N', dic = 'o' ):
        """Function to FIND the Eigenvalue and Eigenstate giving the FOUND Parameters, from before.
        Or in case, given from outside.
        This is a SINGLE EVALUATION. NOT AN OPTIMIZATION. The Parameters are ALREADY OPTIMIZED ALONE. 
        This ISOLATED OPTIMIZATION obviously do not Highlight all the Correlation but a part. 
        Part that can be high or not.
        """
        
        # 1. Taking the Ansatz from the UCC Operator
        self.ansatz = self.ucc_operator
        
        # 2. JW Problem Mapped from Fermionic hamiltonian to Qiskit hamiltonian
        self.hamiltonian = self.JW_problem_mapped
        
        # 3. Estimator given using PAULI Linear Combination
        self.estimator = Estimator()
        
        # 4. Transform the DICTIONARY  value_dict_param_true INTO a LIST of PARAMETERS
        self.param_lst = creation_disperse_array( self.df1['t[i] arcsin tested'].to_dict() , self.ansatz)

        if test == 'Y':
            self.param_lst = creation_disperse_array( dic, self.ansatz)

        
        
        #param_lst = self.creation_disperse_array( param_label )

        t1 = time.time()
        self.eigenvalue = cost_func_vqe( self.param_lst, self.ansatz, self.hamiltonian, self.estimator)
        delta_t = time.time() - t1
        print('Time for the EVALUATION:',delta_t)
        

        print('\nCheck20',self.eigenvalue + self.repulsion )
        
        
        # EIGENVECTOR / EIGENSTATe
        ######################
        #Ope  
        
        """self.optimal_circuit = self.ansatz.assign_parameters( 
            {self.ansatz.parameters[i] : self.param_lst[i] for i in range(len(self.param_lst))} )

        self.optimal_eigenstate = Statevector(self.optimal_circuit)#.draw('latex')

        return self.optimal_eigenstate"""
        
    
    
    def problem_resolution_TAPERING(self, test_tapering='N', lst_tapering = 'o', wf = 'N' ):
        """Function to FIND the Eigenvalue and Eigenstate giving the FOUND Parameters, from before.
        Or in case, given from outside.
        This is a SINGLE EVALUATION. NOT AN OPTIMIZATION. The Parameters are ALREADY OPTIMIZED ALONE. 
        This ISOLATED OPTIMIZATION obviously do not Highlight all the Correlation but a part. 
        Part that can be high or not.
        """
        
        t1 = time.time()
        self.hamiltonian_tapering = self.mapper_tap.map( self.problem.hamiltonian.second_q_op() )
        delta_t = time.time() - t1
        print('time for mapper TAPERING:', delta_t)
        
        #estimator = Estimator()
        # 3. Estimator given using PAULI Linear Combination
        self.estimator = Estimator()
        
        # UCC OPERATORS
        ucc_tap = UCC(
            self.problem.num_spatial_orbitals,
            self.problem.num_particles, 
            self.excitations_type ,
            self.mapper_tap,
            
            initial_state=HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper_tap),)
        
        self.ucc_operator_tapering = ucc_tap
        self.ansatz_tapering = self.ucc_operator_tapering
        
        
        self.param_lst_tapering = list(self.df1['t[i] arcsin tested']) #creation_disperse_array( self.df1['t[i] arcsin tested'].to_dict() ,
                                                         # self.ansatz_tapering)
        
        if test_tapering == 'Y':
            self.param_lst_tapering =  lst_tapering #, self.ansatz_tapering

          
        t1 = time.time()
        self.eigenvalue_tapering = cost_func_vqe( self.param_lst_tapering , 
                                                 self.ansatz_tapering, self.hamiltonian_tapering,
                                                  self.estimator)
           
        
        delta_t = time.time() - t1
        print('time for Evaluation TAPERING:', delta_t, '\n\n')
        
        #repulsion = self.problem.hamiltonian.constants['nuclear_repulsion_energy']
        #core = self.problem.hamiltonian.constants['FreezeCoreTransformer']

        if self.frozen_core =='Y':
            print('\nEigenvalue_Tapering No full opt:',self.eigenvalue_tapering + self.repulsion + self.core, 
                  'h')
            
        else:
            print('\nEigenvalue_Tapering No full opt:',self.eigenvalue_tapering+ self.repulsion, 'h' )
            
        # EIGENVECTOR / EIGENSTATE
        ######################
        #Ope
        
        if wf == 'Y':
        
            self.optimal_circuit_tapering = self.ansatz_tapering.assign_parameters( 
            {self.ansatz_tapering.parameters[i] : self.param_lst_tapering[i] 
             for i in range(len(self.param_lst_tapering))} )

            self.optimal_eigenstate_tapering = Statevector(self.optimal_circuit_tapering)
            #.draw('latex')

            return self.optimal_eigenstate_tapering
        
        else:
            print('\nno wf this time')
    

    

    
    def VQE_run(self, excitation_new, maxrun_opt =10, out_lst = 'N', given_lst = [0]):
        """Function to FIND the Eigenvalue and Eigenstate of the problem.
        It is possible to give or not the initial paramters for a FULL OPTIMIZATION PROCESS.
        """
        
        self.ansatz_vqe = UCC(
            self.problem.num_spatial_orbitals,
            self.problem.num_particles, 
            excitation_new ,
            self.mapper_tap,
            
            initial_state=HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper_tap),)
        
        
        initial_lst = [0]*  self.ansatz_vqe.num_parameters
        
        if out_lst == 'Y':
            initial_lst = given_lst
        
        vqe = VQE( self.estimator, self.ansatz_vqe, L_BFGS_B(maxiter= maxrun_opt), 
              initial_point = initial_lst )
        
        
        self.result = vqe.compute_minimum_eigenvalue(operator= self.hamiltonian_tapering)
        
        print(self.result)
        
        self.eigenvalue_vqe = self.result.optimal_value + self.repulsion
        print(f'Eigenvalue: {self.eigenvalue_vqe }')
        
        # EIGENVECTOR / EIGENSTATE
        ######################
        #Ope
        
        #self.optimal_circuit_vqe = self.ansatz_vqe.assign_parameters( 
        #    {self.ansatz_vqe.parameters[i] : self.result.optimal_point[i] 
        #     for i in range(len( self.result.optimal_point ))} )

        #self.optimal_eigenstate_vqe = Statevector(self.optimal_circuit_vqe)
        #.draw('latex')

        #return self.optimal_eigenstate_vqe
    
    # def FCI PYSCF SIMILAR
    # def Similarity




class Tensor_matrix_toolbox():
    
    def __init__(self, pauli_string_lenght):
        
        self.pauli_string_lenght = pauli_string_lenght
        
        self.matrix_dict = {'I': np.array([[1,0],[0,1]]),
                        'Z': np.array([[1,0],[0,-1]]),
                        'X': np.array([[0,1],[1,0]]),
                        'Y': np.array([[0,-1j],[1j, 0]])}
    
    #################################################################################################################
    # CROWN JEWEL
    #################################################################################################################
    
    def finding_all_ij_pos_in_2_2_matrix(self,   row, col):
        f"""This is MY FIRST CROWN JEWEL Function of the PhD. 
        This is an Independent function able to CONVERT the POSITION ij (row_i, col_j) of a Common Hermitian Matrix A, 
        represented as a linear combination of Pauli Matrices, into: 
        a ordered list (given by a dictionary) of position kl for each indipendent Pauli Matrix constituting a single Pauli String.
        
        RETURN--- dictionary of 2x2 matrices with the Needed ij index to find, by multiplication
        of all these 2x2 Matrices the VALUE of the REAL JW MATRIX at ROW and COL Given.
        
        Remarks:
        In this dictionary the first matrix (also called) Matrix 1 will have an higher weight 
        than the following like in a bit-string. It has the span of 4 times {2**self.pauli_string_lenght/2} x {2**self.pauli_string_lenght/2}. 
        While the second half and so on. 
        This means that a single number in Matrix 1 affects {(2**self.pauli_string_lenght/2) * (2**self.pauli_string_lenght/2)} elements"""
        
        # list of all I matrices (no particular reason why I simple common matrix)
        # given the Number of qubits necessary for JW Mapping
        
        A = np.array([[1,2],[3,1]])
        lst_matrix = [A] * self.pauli_string_lenght
        
        
        
        # matrix one for size of the matrix in the Tensor product order
        pos_matrix = { i: 2**(len(lst_matrix) -i+1) for i in range(1, len(lst_matrix)+1 )}
    
        # just number and matrix correlated
        dict_val_matrix = { i : lst_matrix[i-1] for i in range(1, len(lst_matrix)+1 ) }


        add_i = 0
        add_j = 0
    
        matrix_order_and_elements = {}
    
        for matrix_i in range(1, len(lst_matrix) + 1):

            if row in range(0 + add_i, pos_matrix[matrix_i]// 2 + add_i):
                #print(f'{matrix_i} Row: 0')
                row_temp = 0

            if row in range(pos_matrix[matrix_i]// 2 +add_i, pos_matrix[matrix_i] + add_i):
                #print(f'{matrix_i} Row: 1')
                add_i += pos_matrix[matrix_i]// 2
                row_temp = 1
        
            if col in range(0+ add_j, pos_matrix[matrix_i]// 2+ add_j) :
                #print(f'{matrix_i} Col: 0')
                col_temp = 0
                #print(col_temp)

            if col in range(pos_matrix[matrix_i]// 2+ add_j, pos_matrix[matrix_i]+ add_j):
                #print(f'{matrix_i} Col: 1')
                add_j += pos_matrix[matrix_i]// 2
                col_temp = 1
                #print(col_temp)
        
            #print('\n', dict_val_matrix[matrix_i][row_temp, col_temp])
            #print('\n')
            #print(col_temp)
            matrix_order_and_elements[matrix_i] = (row_temp, col_temp)
        
        dict_matrix_2_2_ij_all = matrix_order_and_elements
        

        
        
        return dict_matrix_2_2_ij_all
    
    
    ###################################################################################################
    ###################################################################################################
        
       

    def energy_or_exc_element_ij(self, row, col, lin_comb_pauli_strings):
        """This is MY SECOND CROWN JEWEL Function of the PhD. 
        This is an Dependent function able to EXTRACT the VALUE x at the POSITION ij of an Hermitian Matrix A, 
        represented as a linear combination of Pauli Matrices. This function employs the previous function to obtain the Ordered list 
        of position kl of each independent Pauli Matrix.
        The product of each Pauli string for its coefficient and sum with all the other Pauli strings will give me the VALUE x at the 
        POSITION ij.
        
        It takes:
        1. row i of the Hermitian Matrix
        2. column j of the Hermitian Matrix
        3. the Linear Combination of Pauli strings reported as a SparsePauliOp
        
        RETURN --- single value
        
        REMARKS --- Steps in the Algorithm:
        1. The loop starts with taking the First all the Pauli word in the Linear Combination of PAULI STRINGS 
        2. This Pauli Word, called pauli_word, will have a certain lenght that is equal to the number of qubits
        3. Here starts a second loop. For each individual Pauli Matrix, called pauli_word[i], in a Pauli word, 
        I get their respective element kl given by the first function, calling dict_matrix_2_2_ij_all[ i +1 ]
        4. This data is stored
        5. This data then is Multiplied by the coefficient belonging to the Pauli string
        6. Finally I sum everything to get the single value x
        """
        
        # It creates the Dictionary of 2x2 Matrix Positions to find the value requested
        # in the NxN real Matrix as tensor
        dict_matrix_2_2_ij_all = self.finding_all_ij_pos_in_2_2_matrix( row, col)
    
        paulis2 =  lin_comb_pauli_strings.paulis #self.JW_problem_mapped.paulis
        coeffs2 =  lin_comb_pauli_strings.coeffs #self.JW_problem_mapped.coeffs
        


        all_matrix_i_lst = []

        for mat_i in range(len( paulis2 )):
    
            pauli_word = list(paulis2[mat_i].settings['data']) # it creates the list of I and Z from Pauli e.g IIZX
            #print(pauli_word)
            matrix_i_value = 1
            for i in range( self.pauli_string_lenght ):
            
                # DATA OF A SINGLE LETTER IN MATRIX
            
                # pauli_word[i] just I or Z --> Gives Array --> gives row_i and col_j
                # for that peculiar Position in Pauli
                data = self.matrix_dict[ pauli_word[i] ][ dict_matrix_2_2_ij_all[ i +1 ]]
                #print(data)
                matrix_i_value *=data
        
            # DATA of the Overall STRING MATRIX MULTIPLIED by ITS WEIGHT
            matrix_i_value *= coeffs2[mat_i]
            #print('\n', matrix_i_value, '\n')
        
            # Collection of ALL MATRIX I Multiplied by WEIGHT
    
            all_matrix_i_lst.append(matrix_i_value)
    
    
        # Energy of that Configuration in Interaction with Itself of Another one
        energy_elem_ij = sum(all_matrix_i_lst)
        
        #######################
        ######################if self.full_print == 'Y':
        # print(len(all_matrix_i_lst))
        # print('\n', '*'*75, '\nBANG energy_elem_ij:', f'\t{energy_elem_ij} h', '\n', '*'*75)

        return energy_elem_ij

    

def cost_func_vqe(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return cost



def dot_matrix(elem_mat, sign, t):
    
    H = np.array([[elem_mat[0] , elem_mat[1] ], 
                  [elem_mat[1] , elem_mat[2] ] ])  
    
    
    
    t_pos = abs(t) # t positive
    U_pos = np.array([[np.cos(t_pos)           , np.sin(t_pos)*sign ], 
                      [np.sin(t_pos)*sign*-1, np.cos(t_pos)         ] ])
    
    matrix_ii_t_pos = np.dot( U_pos.transpose(), np.dot( H, U_pos ) )[0,0]
    
    
    t_neg = abs(t) * -1 # t negative
    U_neg = np.array([[np.cos(t_neg )           , np.sin(t_neg )*sign ], 
                      [np.sin(t_neg )*sign*-1, np.cos(t_neg )         ] ])
    
    matrix_ii_t_neg = np.dot( U_neg.transpose(), np.dot( H, U_neg ) )[0,0]
    
    if matrix_ii_t_pos < matrix_ii_t_neg:
        #print('positive param', t_pos, 'jj')
        
        return t_pos
        
    else:
        #print('negative param', t_neg, 'kk')
        
        return t_neg
    
    #value = np.dot(np.array([1,0]).transpose(), np.dot(matrix , np.array([1,0]) ) )
    #a, b = np.linalg.eig(matrix)
    
    #return matrix#, U # value
    
    
def creation_disperse_array(resized_dict, ansatz):
    full_disperse_array = [0]*ansatz.num_parameters
    
    for i in resized_dict.keys():
        full_disperse_array[i] = resized_dict[i]
    
    #print(full_disperse_array)
    return full_disperse_array

