#ifndef _LEADELE_LINEAR_MODEL_HPP_
#define _LEADELE_LINEAR_MODEL_HPP_

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <functional>

#include "../data/mnist.hpp"

namespace leadele {
struct LinearModel {
  struct parameter_k {
    parameter_k(int size): weights(size, 0.0), shift(0) {
      // auto gen = []() -> double {
      //   auto now = std::chrono::system_clock::now();
      //   std::mt19937 mt(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count()); // Standard mersenne_twister_engine seeded with rd()
      //   std::uniform_real_distribution<double> dist(0.0, 1.0);
      //   return dist(mt);
      // };
      // std::generate(weights.begin(), weights.end(), gen);
    }
    std::vector<double> weights;
    double shift;
  };

public:
  LinearModel(int nbr_input_nodes, int nbr_output_nodes, int batch_size, double _eta)
    : N(nbr_input_nodes), K(nbr_output_nodes), B(batch_size), params(nbr_output_nodes, nbr_input_nodes), eta(_eta) {}

  /**
   * @brief Updates the parameters, using the training data set and using the gradient descent method.
   * @param training_set Training dataset, a std::vector<leadele::Image>.
   */
  void gradient_descent(leadele::Data const& training_set) {
    {
      for (unsigned int b = 0; b < training_set.size()/B; ++b) {
        // Compute answers to batch of size B
        auto begin = std::next(training_set.begin(), b*B); // Beginning of batch b
        std::vector<std::vector<double>> responses = compute_answers(begin, B);

        // Update the parameters w_ki and b_k
        update_parameters(training_set, responses, b);
      }
    }
  }

  void gradient_descent2(leadele::Data const& training_set) {
    {
      for (unsigned int b = 0; b < training_set.size()/B; ++b) {
        // Compute answers to batch of size B
        auto begin = std::next(training_set.begin(), b*B); // Beginning of batch b
        std::vector<std::vector<double>> responses = compute_answers2(begin, B);

        // Update the parameters w_ki and b_k
        update_parameters2(training_set, responses, b);
      }
    }
  }

  double compute_accuracy(leadele::Data const& test_set) const {
    return std::transform_reduce(test_set.begin(), test_set.end(), 0.0,
                                 std::plus<>(),
    [&](leadele::Image const& img) -> double {
      auto answer = compute_answer(img);
      int ans_lbl = std::distance(answer.begin(), std::max_element(answer.begin(), answer.end()));
      return (img.label==ans_lbl)? 1.0: 0.0;
    })/test_set.size();
  }

  /**
   * @brief Compute the Loss Function over the test dataset.
   * @param test_set A testing dataset, for validation. Must be different from the training dataset.
   * @return The evaluation of the Loss Function (a minus log-likelihood calculation).
   */
  double compute_loss_function(leadele::Data const& test_set) const {
    std::vector<std::vector<double>> responses = compute_answers(test_set.cbegin(), test_set.size());
    return -1.0/static_cast<double>(test_set.size()) * std::transform_reduce(responses.begin(), responses.end(), test_set.begin(), 0.0,
           std::plus<>(), [&](std::vector<double> const& y_m, leadele::Image const img) -> double { return std::log(y_m[img.label]);; });
  }

  double compute_loss_function2(leadele::Data const& test_set) const {
    std::vector<std::vector<double>> logits(test_set.size(), std::vector<double>(K));
    for (int i = 0; i < logits.size(); ++i) {
      auto& a_m = logits[i];
      leadele::Image const& img = test_set[i];

      for (int k = 0; k < K; ++k) {
        // Inner product
        a_m[k] = params[k].shift;
        for (int l = 0; l < N; ++l) {
          a_m[k] += params[k].weights[l]*img[l];
        }
      }
    }

    // Normalize, using sigma function, the logits to compute the responses y_k(m), 1 <= k <= K, 1 <= m <= B
    std::vector<std::vector<double>> responses(test_set.size(), std::vector<double>(K));
    for (int i = 0; i < responses.size(); ++i) {
      auto& a_m = logits[i];
      auto& y_m = responses[i];
      double div {};
      for (int k = 0; k < K; ++k) {
        div += std::exp(a_m[k]);
      }
      for (int k = 0; k < K; ++k) {
        y_m[k] = std::exp(a_m[k])/div;
      }
    }

    double norm {-1.0/static_cast<double>(test_set.size())};
    double L {};
    for (int i = 0; i < responses.size(); ++i) {
      auto& y_m = responses[i];
      auto& img = test_set[i];

      L += std::log(y_m[img.label]);
    }

    return norm * L;
  }

  /**
   * @brief Compute the answer of the network to an image.
   * @param img A vectorized image, in gray scale, with values in [0, 1].
   * @return A vector of size K (number of classes), where value of element k gives belonging (1 if belongs to class k, 0 otherwise).
   */
  std::vector<double> compute_answer(leadele::Image const& img) const {
    std::vector<double> logits(K);

    // For parameter k, compute the logit a_k(m)
    std::transform(params.begin(), params.end(), logits.begin(), [&](parameter_k const& p_k) -> double {
      return std::inner_product(p_k.weights.begin(), p_k.weights.end(), img.begin(), p_k.shift);
    });

    // Normalize, using sigma function, the logits to compute the responses y_k(m), 1 <= k <= K, 1 <= m <= B
    std::vector<double> response(K);
    double div = std::accumulate(logits.begin(), logits.end(), 0.0, [&](double left, double right) -> double { return std::move(left) + std::exp(right); });
    std::transform(logits.begin(), logits.end(), response.begin(), [&](double logit) -> double {
      return sigma(logit, div);
    });

    return response;
  }

  std::vector<double> compute_answer2(leadele::Image const& img) const {
    std::vector<double> a_m(K);

    for (int k = 0; k < K; ++k) {
      // Inner product
      a_m[k] = params[k].shift;
      for (int l = 0; l < N; ++l) {
        a_m[k] += params[k].weights[l]*img[l];
      }
    }

    // Normalize, using sigma function, the logits to compute the responses y_k(m), 1 <= k <= K, 1 <= m <= B
    std::vector<double> y_m(K);
    double div {};
    for (int k = 0; k < K; ++k) {
      div += std::exp(a_m[k]);
    }
    for (int k = 0; k < K; ++k) {
      y_m[k] = std::exp(a_m[k])/div;
    }

    return y_m;
  }

  // leadele::Image compute_reverse_answer(std::vector<double> const& classes) const {
  //   leadele::Image
  // }

  /**
   * @brief Update the parameters (the weights \^$w_{ki}\$ and the shifts \$b_k\$), using gradient descent.
   * @param training_set The training dataset.
   * @param responses The array of responses of the output nodes from the training dataset.
   * @param b The batch index (for Batch or Stochastic Gradient Descent).
   */
  void update_parameters(leadele::Data const& training_set, std::vector<std::vector<double>> responses, unsigned int b) {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < N; ++i) {
        auto begin = std::next(training_set.begin(), b*B); // Beginning of a batch
        params[k].weights[i] -= eta/B * std::transform_reduce(responses.begin(), responses.end(), begin, 0.0, std::plus<>(),
                                [&](std::vector<double> const& y_m, leadele::Image const& img) -> double { return static_cast<double>(img[i])*(y_m[k]-((img.label==k)? 1: 0)); });
      }
      auto begin = std::next(training_set.begin(), b*B); // Beginning of a batch
      params[k].shift -= eta/B * std::transform_reduce(responses.begin(), responses.end(), begin, 0.0, std::plus<>(),
                         [&](std::vector<double> const& y_m, leadele::Image const& img) -> double { return (y_m[k]-((img.label==k)? 1: 0)); });
    }
  }

  void update_parameters2(leadele::Data const& training_set, std::vector<std::vector<double>> responses, unsigned int b) {
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < N; ++i) {
        double acc {};
        for (int r = 0; r < responses.size(); ++r) {
          auto& y_m = responses[r];
          auto& img = training_set[b*B + r];

          acc += img[i]*(y_m[k]-((img.label==k)? 1: 0));
        }

        params[k].weights[i] -= -1*eta/static_cast<double>(B) * acc;
      }
      double acc {};
      for (int r = 0; r < responses.size(); ++r) {
        auto& y_m = responses[r];
        auto& img = training_set[b*B + r];

        acc += (y_m[k]-((img.label==k)? 1: 0));
      }

      params[k].shift -= -1*eta/static_cast<double>(B) * acc;
    }
  }

  /**
   * @brief Compute the answer of the network to the input nodes.
   * @param begin Constant iterator to the intput dataset (can be training or test).
   * @param size  The size of the batch.
   * @return The matrix (size size x K) of answers y_k(m) to the batch input.
   */
  std::vector<std::vector<double>> compute_answers(leadele::Data::const_iterator begin, int size) const {
    std::vector<std::vector<double>> responses(size, std::vector<double>(K));
    std::transform(begin, std::next(begin, size), responses.begin(), [&](leadele::Image const& img) -> std::vector<double> { return compute_answer(img); });
    return responses;
  }

  std::vector<std::vector<double>> compute_answers2(leadele::Data::const_iterator& begin, int size) const {
    // Normalize, using sigma function, the logits to compute the responses y_k(m), 1 <= k <= K, 1 <= m <= B
    std::vector<std::vector<double>> responses(size, std::vector<double>(K));
    auto it = begin;
    for (int i = 0; i < responses.size(); ++i, ++it) {
      responses[i] = compute_answer(*it);
    }

    return responses;
  }

  /**
   * @brief Sigma function for normalizing the output responses from the logits.
   * @param logit One of the logit.
   * @param div Normalization factor: \$\sum_{k=1}^K\exp{a_k}\$.
   * @return The normalized logit value.
   */
  double sigma(double logit, double div) const {
    return std::exp(logit)/div;
  }

  int N; /**< Number of input nodes. */
  int K; /**< Number of output nodes. */
  int B; /**< Batch size: number of training items in a batch. */

  std::vector<parameter_k> params; /**< Matrix of connections from input nodes to output nodes. */
  double eta; /**< Learning rate. */

  // static std::execution::sequenced_policy constexpr def_exec_policy = std::execution::seq;
};
}

#endif // _LEADELE_LINEAR_MODEL_HPP_
