#ifndef _LEADELE_DATA_HPP_
#define _LEADELE_DATA_HPP_

#include <vector>

#include "image.hpp"

namespace leadele {
/**
 * @brief A basic Data, contains the images to be classified (either training or test set).
 */
struct Data: public std::vector<Image> {
  int32_t nbr_images ;
  int32_t nbr_rows   ;
  int32_t nbr_columns;
  int32_t nbr_labels;
};
}

#endif // _LEADELE_DATA_HPP_
