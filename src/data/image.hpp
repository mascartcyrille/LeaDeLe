#ifndef LEADELE_IMAGE_HPP_
#define LEADELE_IMAGE_HPP_

#include <vector>

namespace leadele {
/**
 * @brief A generic Image.
 *
 * An Image is an item to be classified: a number, a picture, etc.
 * This is a simple decorator over a std::vector<double>, which adds some information about the diemnsions of the image, as well as the label to be learnt.
 *
 * In the case of a 2-D picture, the linearilization is column major:
 * 123
 * 456 -> 123456789
 * 789
 */
struct Image: public std::vector<double> {
public:
  /**
   * @brief Simple constructor.
   * @param _rows Number of rows.
   * @param _columns Number of columns.
   * @param _label Label of the image.
   */
  Image(int _rows, int _columns, int _label): std::vector<double>(_rows * _columns), rows(_rows), columns(_columns), label(_label) {}

  int rows; /**< Number of rows of the image. */
  int columns; /**< Number of columns of the image. */
  int label; /**< Label of the image. */
};
}

#endif // LEADELE_IMAGE_HPP_
