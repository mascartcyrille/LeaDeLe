#ifndef MNIST_HPP
#define MNIST_HPP

#include <cerrno>
#include <cstring>    // For std::strerror
#include <fstream>
#include <iostream>

#include "image.hpp"
#include "data.hpp"

namespace leadele {
struct MNIST: public Data {
  /**
   * @brief Decodes and stores the images and labels from the MNIST dataset for machine learning.
   * @param images_file
   * @param labels_file
   */
  MNIST(std::string const& images_file, std::string const& labels_file): Data() {
    {
      // Open data files
      std::ifstream ifs_images(images_file, std::ios::binary);
      std::ifstream ifs_labels(labels_file, std::ios::binary);

      if (!ifs_images.is_open() || !ifs_images.is_open()) {
        std::cerr << "Error while opening file " << images_file << " or file " << labels_file << ": " << std::strerror(errno) << std::endl;
        exit(-1);
      }

      // The two magic numbers for file decoding validation
      int32_t img_magic_nbr;
      int32_t lbl_magic_nbr;

      // Read the headers of the image data file
      unsigned char s_img_nbr_columns[4] {}; ifs_images.read((char*)&s_img_nbr_columns, sizeof(int32_t));
      unsigned char s_img_nbr_rows[4]    {}; ifs_images.read((char*)&s_img_nbr_rows,    sizeof(int32_t));
      unsigned char s_img_nbr_images[4]  {}; ifs_images.read((char*)&s_img_nbr_images,  sizeof(int32_t));
      unsigned char s_img_magic_nbr[4]   {}; ifs_images.read((char*)&s_img_magic_nbr,   sizeof(int32_t));

      // Little endian --> Big endian transformation
      img_magic_nbr = (int)s_img_magic_nbr[3]   | (int)s_img_magic_nbr[2]<<8   | (int)s_img_magic_nbr[1]<<16   | (int)s_img_magic_nbr[0]<<24;
      nbr_images    = (int)s_img_nbr_images[3]  | (int)s_img_nbr_images[2]<<8  | (int)s_img_nbr_images[1]<<16  | (int)s_img_nbr_images[0]<<24;
      nbr_rows      = (int)s_img_nbr_rows[3]    | (int)s_img_nbr_rows[2]<<8    | (int)s_img_nbr_rows[1]<<16    | (int)s_img_nbr_rows[0]<<24;
      nbr_columns   = (int)s_img_nbr_columns[3] | (int)s_img_nbr_columns[2]<<8 | (int)s_img_nbr_columns[1]<<16 | (int)s_img_nbr_columns[0]<<24;

      // Validate the transformation
      if (img_magic_nbr != 2051) {
        std::cerr << "Wrong magic number: " << img_magic_nbr << " instead of 2051." << std::endl;
        exit(-1);
      }

      // Read the headers of the label data file
      unsigned char s_lbl_magic_nbr[4]   {}; ifs_labels.read((char*)&s_lbl_magic_nbr,   sizeof(int32_t));
      unsigned char s_lbl_nbr_labels[4]  {}; ifs_labels.read((char*)&s_lbl_nbr_labels,  sizeof(int32_t));

      // Little endian --> Big endian transformation
      lbl_magic_nbr   = (int)s_lbl_magic_nbr[3]  | (int)s_lbl_magic_nbr[2]<<8  | (int)s_lbl_magic_nbr[1]<<16  | (int)s_lbl_magic_nbr[0]<<24;
      nbr_labels  = (int)s_lbl_nbr_labels[3] | (int)s_lbl_nbr_labels[2]<<8 | (int)s_lbl_nbr_labels[1]<<16 | (int)s_lbl_nbr_labels[0]<<24;

      // Validate the transformation
      if (lbl_magic_nbr != 2049) {
        std::cerr << "Wrong magic number: " << lbl_magic_nbr << " instead of 2049." << std::endl;
        exit(-1);
      }

      // Validate that there is the same number of labels and images
      if (nbr_images != nbr_labels) {
        std::cerr << "Different number of images and labels: " << nbr_images << " =!= " << nbr_labels << std::endl;
        exit(-1);
      }


      this->reserve(nbr_images);
      for (auto img = 0; img < nbr_images; ++img) {
        unsigned char new_label;
        ifs_labels.read((char*)&new_label, sizeof(unsigned char));
        Image new_image(nbr_rows, nbr_columns, static_cast<int>(new_label));
        for (int r = 0; r < nbr_rows; ++r) {
          for (int c = 0; c < nbr_columns; ++c) {
            unsigned char pixel;
            ifs_images.read((char*)&pixel, sizeof(unsigned char));
            new_image[c + r*nbr_columns] = static_cast<double>(pixel)/255.;
          }
        }
        this->push_back(new_image);
      }

      // Close the data files
      ifs_images.close();
      ifs_labels.close();
    }
  }
};
}

#endif // MNIST_HPP
