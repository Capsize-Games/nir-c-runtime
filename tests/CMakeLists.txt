function(nir_add_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE nir_c_runtime)
  add_test(NAME ${name} COMMAND ${name}
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

nir_add_test(test_json)
nir_add_test(test_ops)
nir_add_test(test_conv_net)
nir_add_test(test_fixed_event)
nir_add_test(test_quantized)
