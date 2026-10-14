add_executable(mnist_train mnist_train.c)
target_link_libraries(mnist_train PRIVATE nir_c_runtime)

add_test(
  NAME mnist_train_example
  COMMAND mnist_train
    ${CMAKE_SOURCE_DIR}/examples/mnist_tiny_graph.json
    ${CMAKE_SOURCE_DIR}/tests/fixtures/mnist_tiny_images.idx
    ${CMAKE_SOURCE_DIR}/tests/fixtures/mnist_tiny_labels.idx
    ${CMAKE_BINARY_DIR}/mnist_tiny_checkpoint.json
    20
)
