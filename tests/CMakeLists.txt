function(nir_add_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE nir_c_runtime)
  add_test(NAME ${name} COMMAND ${name}
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

nir_add_test(test_json)
nir_add_test(test_graph)
nir_add_test(test_ops)
nir_add_test(test_relu)
nir_add_test(test_conv_net)
nir_add_test(test_neuron_kinds)
nir_add_test(test_fixed_event)
nir_add_test(test_quantized)
nir_add_test(test_workspace)
nir_add_test(test_tensor)
nir_add_test(test_error_format)
nir_add_test(test_train)
nir_add_test(test_backward)
nir_add_test(test_stdp_graph)
nir_add_test(test_init)
nir_add_test(test_init_graph)
nir_add_test(test_schedule)
nir_add_test(test_loss)
nir_add_test(test_rng)
nir_add_test(test_encode)
nir_add_test(test_data)
nir_add_test(test_checkpoint)
nir_add_test(test_minibatch)
nir_add_test(test_tape)
nir_add_test(test_driver)

# OOM sweep includes src/json.c directly, so it must not link the library
# (that would duplicate the json_* symbols); it only needs the headers.
add_executable(test_json_oom test_json_oom.c)
target_include_directories(test_json_oom PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_json_oom COMMAND test_json_oom
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_graph_bias test_graph_bias.c)
target_link_libraries(test_graph_bias PRIVATE nir_c_runtime)
foreach(case affine-short affine-long affine-empty affine-invalid affine-exact
             conv-short conv-long conv-empty conv-invalid conv-exact
             conv-omitted conv-null
             conv1d-short conv1d-long conv1d-empty conv1d-invalid conv1d-exact
             conv1d-omitted conv1d-null)
  add_test(NAME test_graph_bias_${case} COMMAND test_graph_bias ${case})
endforeach()
