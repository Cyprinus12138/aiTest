import struct
import numpy as np


class MNISTDecoder:
    def __init__(self, ubyte_file):
        self.fmt_1 = ">2i"
        self.fmt_3 = ">4i"
        self.bin_content = open(ubyte_file, 'rb').read()
        file_extension = ubyte_file.split('.')[-1]
        if file_extension == 'idx1-ubyte':
            self.idx1()
        elif file_extension == 'idx3-ubyte':
            self.idx3()
        else:
            raise TypeError("Unavailable file type")

    def idx1(self):
        offset = 0
        magic_number, number_of_items = struct.unpack_from(self.fmt_1, self.bin_content, offset)
        offset += struct.calcsize(self.fmt_1)
        fmt_label = '>B'
        labels = np.empty(number_of_items)
        for i in range(number_of_items):
            labels[i] = struct.unpack_from(fmt_label, self.bin_content, offset)
            offset += struct.calcsize(fmt_label)
        return labels

    def idx3(self):
        offset = 0
        magic_number, number_of_images, rows, columns = struct.unpack_from(self.fmt_3, self.bin_content, offset)
        offset += struct.calcsize(self.fmt_3)
        image_size = rows * columns
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty(number_of_images)
        for i in range(number_of_images):
            image = np.array(struct.unpack_from(fmt_image, self.bin_content, offset)).reshape((rows, columns))
            offset += struct.calcsize(fmt_image)
            images[i] = image
        return images




