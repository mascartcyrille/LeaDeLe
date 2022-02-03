#include <chrono>
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip>

#include "../../src/data/image.hpp"
#include "../../src/data/data.hpp"
#include "../../src/data/rounding.hpp"
#include "../../src/learning/linear-model.hpp"

void cout_answers_and_parameters(leadele::LinearModel const& lm, leadele::ROUNDING const& test_set, int iter) {
  std::cout << "##### Compute answer (" << 1+iter << " iterations) #####" << std::endl;
  for (int i = 0; i < 10; ++i) {
    auto tmp = lm.compute_answer(test_set[i]);
    std::cout << "Answer " << std::distance(tmp.begin(), std::max_element(tmp.begin(), tmp.end())) << " is " << (test_set[i].label == std::distance(tmp.begin(),
              std::max_element(tmp.begin(), tmp.end()))) << " on image " << i << ": label " << test_set[i].label << " ";
    std::copy(tmp.begin(), tmp.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << " | sum = " << std::accumulate(tmp.begin(), tmp.end(), 0.0) << std::endl;
  }

  std::cout << std::endl << "##### Display parameters matrix (" << 1+iter << " iterations) #####" << std::endl;
  for (int k = 0; k < lm.params.size(); ++k) {
    std::cout << "b_" << k << "|w_{" << k << ",i}: " << lm.params[k].shift << " | ";
    std::copy(lm.params[k].weights.begin(), lm.params[k].weights.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
  }
}

int main(void) {
  // leadele::MNIST train_set("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
  // leadele::MNIST test_set("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

  std::random_device rd{};
  std::mt19937_64 prng{rd()};
  leadele::ROUNDING train_set(60'000, prng);
  leadele::ROUNDING test_set(10'000, prng);

  int const nbr_input_nodes = train_set.nbr_columns*train_set.nbr_rows;
  int const nbr_output_nodes = 10;
  int const batch_size = 1;
  int const iterations = 3;

  double const learning_rate = 0.1;
  leadele::LinearModel lm(nbr_input_nodes, nbr_output_nodes, batch_size, learning_rate);

  cout_answers_and_parameters(lm, test_set, -1);
  std::cout << std::endl << "##### Accuracy (no iteration): " << lm.compute_accuracy(test_set) << " #####" << std::endl;

  for (int t = 0; t < train_set.size(); ++t) {
    lm.update_parameters(train_set, std::vector<std::vector<double>> {{lm.compute_answer(train_set[t])}}, t);
  }

  std::cout << std::endl;

  cout_answers_and_parameters(lm, test_set, 0);
  double acc = lm.compute_accuracy(test_set);
  std::cout << std::endl << "##### Accuracy (1 iteration): " << acc << " #####" << std::endl;

  bool cntn = false;
  int iter {};
  double acc_new;
  do {
    for (int t = 0; t < train_set.size(); ++t) {
      lm.update_parameters(train_set, std::vector<std::vector<double>> {{lm.compute_answer(train_set[t])}}, t);
    }

    acc_new = lm.compute_accuracy(test_set);
    if (acc < acc_new) {
      cntn = true;
      acc = acc_new;
    } else {
      cntn = false;
    }
  } while (iter++ < iterations && cntn);
  std::cout << std::endl;

  cout_answers_and_parameters(lm, test_set, iter);
  std::cout << std::endl << "##### Accuracy (" << 1+iter << " iterations): " << acc << " vs. " << acc_new << " #####" << std::endl;

  return 0;
}
