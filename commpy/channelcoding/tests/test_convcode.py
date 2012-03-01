
from numpy import array
from numpy.testing import assert_equal, assert_array_equal, run_module_suite

from commpy.channelcoding.convcode import Trellis

# Test Case 1: 1/2 - RSC
memory_1 = array([2])
g_matrix_1 = array([[04, 05]])
feedback_1 = 07
code_type_1 = 'rsc'
next_state_table_1 = array([[0, 2], 
                            [2, 0], 
                            [3, 1], 
                            [1, 3]])
output_table_1 = array([[0, 3], 
                        [0, 3], 
                        [1, 2], 
                        [1, 2]]) 

# =============================================================================
# Test Case 2:
memory_2 = array([2])
g_matrix_2 = array([[05, 07]])
feedback_2 = 00
code_type_2 = 'default'
next_state_table_2 = array([[0, 2], 
                            [0, 2], 
                            [1, 3], 
                            [1, 3]])
output_table_2 = array([[0, 3],
                        [3, 0],
                        [1, 2],
                        [2, 1]])

# =============================================================================
# Test Case 3:
memory_3 = array([3])
g_matrix_3 = array([[013, 017]])
feedback_3 = 00
code_type_3 = 'default'
next_state_table_3 = array([[0, 4], 
                            [0, 4],
                            [1, 5],
                            [1, 5],
                            [2, 6],
                            [2, 6],
                            [3, 7],
                            [3, 7]])
output_table_3 = array([[0, 3],
                        [3, 0],
                        [3, 0],
                        [0, 3],
                        [1, 2],
                        [2, 1],
                        [2, 1],
                        [1, 2]])

# =============================================================================
# Test Case 4:

memory_4 = array([1, 2, 2])
g_matrix_4 = array([[03, 02, 02, 03],
                    [02, 03, 00, 07],
                    [00, 02, 05, 05]])
feedback_4 = 00
code_type_4 = 'default'
number_inputs_4 = 8
number_states_4 = 32
next_state_table_4 = array([[0, 2, 8, 10, 16, 18, 24, 26],
                            [0, 2, 8, 10, 16, 18, 24, 26],
                            [1, 3, 9, 11, 17, 19, 25, 27],
                            [0, 2, 8, 10, 16, 18, 24, 26],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [0, 16, 4, 20, 1, 17, 5, 21],
                            [0, 16, 4, 20, 1, 17, 5, 21],
                            [0, 16, 4, 20, 1, 17, 5, 21],
                            [0, 16, 4, 20, 1, 17, 5, 21],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [2, 18, 6, 22, 3, 19, 7, 23],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [8, 24, 12, 28, 9, 25, 13, 29],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31],
                            [10, 26, 14, 30, 11, 27, 15, 31]])

output_table_4 = array([[0, 3, 1, 2, 17, 14, 16, 15],
                        [11, 12, 10, 13, 6, 5, 7, 4],
                        [5, 6, 4, 7, 12, 11, 13, 10],
                        [14, 17, 15, 16, 3, 0, 2, 1],
                        [15, 16, 14, 17, 2, 1, 3, 0],
                        [4, 7, 5, 6, 13, 10, 12, 11],
                        [10, 13, 11, 12, 7, 4, 6, 5],
                        [1, 2, 0, 3, 16, 15, 17, 14],
                        [3, 0, 2, 1, 14, 17, 15, 16],
                        [12, 11, 13, 10, 5, 6, 4, 7],
                        [6, 5, 7, 4, 11, 12, 10, 13],
                        [17, 14, 16, 15, 0, 3, 1, 2],
                        [16, 15, 17, 14, 1, 2, 0, 3],
                        [7, 4, 6, 5, 10, 13, 11, 12],
                        [13, 10, 12, 11, 4, 7, 5, 6],
                        [2, 1, 3, 0, 15, 16, 14, 17],
                        [4, 7, 5, 6, 13, 10, 12, 11],
                        [15, 16, 14, 17, 2, 1, 3, 0],
                        [1, 2, 0, 3, 16, 15, 17, 14],
                        [10, 13, 11, 12, 7, 4, 6, 5],
                        [11, 12, 10, 13, 6, 5, 7, 4],
                        [0, 3, 1, 2, 17, 14, 16, 15],
                        [14, 17, 15, 16, 3, 0, 2, 1],
                        [5, 6, 4, 7, 12, 11, 13, 10],
                        [7, 4, 6, 5, 10, 13, 11, 12],
                        [16, 15, 17, 14, 1, 2, 0, 3],
                        [2, 1, 3, 0, 15, 16, 14, 17],
                        [13, 10, 12, 11, 4, 7, 5, 6],
                        [12, 11, 13, 10, 5, 6, 4, 7],
                        [3, 0, 2, 1, 14, 17, 15, 16],
                        [17, 14, 16, 15, 0, 3, 1, 2],
                        [6, 5, 7, 4, 11, 12, 10, 13]])

# =============================================================================
# Test Case 5:
memory_5 = array([1, 1])
g_matrix_5 = array([[03, 02, 03],
	                [02, 01, 01]])
feedback_5 = 00
code_type_5 = 'default'
number_inputs_5 = 4
number_states_5 = 4
next_state_table_5 = array([[0, 1, 2, 3],
                            [0, 1, 2, 3],
                            [0, 1, 2, 3],
                            [0, 1, 2, 3]])
output_table_5 = array([[0, 4, 7, 3],
                        [3, 7, 4, 0],
                        [5, 1, 2, 6],
                        [6, 2, 1, 5]])




class TestTrellis:

    def test_functional(self):
        trellis_1 = Trellis(memory_1, g_matrix_1, feedback_1, code_type_1)
        assert_array_equal(trellis_1.next_state_table, next_state_table_1)
        assert_array_equal(trellis_1.output_table, output_table_1)
        
        trellis_2 = Trellis(memory_2, g_matrix_2, feedback_2, code_type_2)
        assert_array_equal(trellis_2.next_state_table, next_state_table_2)
        assert_array_equal(trellis_2.output_table, output_table_2)

        trellis_3 = Trellis(memory_3, g_matrix_3, feedback_3, code_type_3)
        assert_array_equal(trellis_3.next_state_table, next_state_table_3)
        assert_array_equal(trellis_3.output_table, output_table_3)
        
        trellis_5 = Trellis(memory_5, g_matrix_5, feedback_5, code_type_5)
        assert_array_equal(trellis_5.next_state_table, next_state_table_5)
        assert_array_equal(trellis_5.output_table, output_table_5)

if __name__ == "__main__":
    run_module_suite()
