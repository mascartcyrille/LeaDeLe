#ifndef _LEADELE_ROUNDING_HPP_
#define _LEADELE_ROUNDING_HPP_

#include <random>

#include "image.hpp"
#include "data.hpp"

namespace leadele {
struct ROUNDING: public Data {
  ROUNDING(int size, std::mt19937_64& prng): Data() {
    nbr_images = size;
    nbr_rows = 1;
    nbr_columns = 1;
    nbr_labels = size;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
      double rdn {dist(prng)*10};
      Image img(1, 1, (int)(rdn));
      img[0] = rdn;
      this->push_back(img);
    }
  }
};
}

#endif // _LEADELE_ROUNDING_HPP_
