
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
memory_4 = array([1, 1])
g_matrix_4 = array([[03, 02, 03],
	                [02, 01, 01]])
feedback_4 = 00
code_type_4 = 'default'
number_inputs_4 = 4
number_states_4 = 4
next_state_table_4 = array([[0, 2, 1, 3],
                          [0, 2, 1, 3],
                          [0, 2, 1, 3],
                          [0, 2, 1, 3]])
output_table_4 = array([[0, 4, 7, 3],
                        [5, 1, 2, 3],
                        [3, 7, 4, 0],
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
        
        trellis_4 = Trellis(memory_4, g_matrix_4, feedback_4, code_type_4)
        assert_array_equal(trellis_4.next_state_table, next_state_table_4)
        assert_array_equal(trellis_4.output_table, output_table_4)

if __name__ == "__main__":
    run_module_suite()
