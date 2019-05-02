import unittest
import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, '../')
from match_channels import match_channels, load_files

class TestMatchChannels(unittest.TestCase):
    def setUp(self):
        self.file1 = './TEST_FILE_1'
        self.file2 = './TEST_FILE_2'

        # Batch size of 1, 4 channels, each with 16x16 data
        tensor1 = torch.zeros([1,4,16,16])
        tensor1.bernoulli_()

        # Permute channels of tensor1
        tensor2 = tensor1[:,torch.LongTensor([1,3,0,2]),:,:]

        self.data1 = [[tensor1]]
        self.data2 = [[tensor2]]

        file = open(self.file1, "wb")
        pickle.dump(self.data1, file)
        file.close()

        file = open(self.file2, "wb")
        pickle.dump(self.data2, file)
        file.close


    def tearDown(self):
        os.remove(self.file1)
        os.remove(self.file2)
        if os.path.exists('./TEST_FILE_2_matched'):
            os.remove('./TEST_FILE_2_matched')


    def test_load_files(self):
        # Tests load_files(file1, file2)
        data1, data2 = load_files(self.file1, self.file2)

        for i in range(len(data1)):
            for j in range(len(data1[0])):
                self.assertTrue(torch.all(torch.eq(data1[i][j],self.data1[i][j])))

        for i in range(len(data2)):
            for j in range(len(data2[0])):
                self.assertTrue(torch.all(torch.eq(data2[i][j],self.data2[i][j])))


    def test_match_channels_normal(self):
        # Tests match_channels(..., mode=normal)
        _, d2 = load_files(self.file1, self.file2)

        match_channels(self.file1, self.file2, 'normal')
        with open('./TEST_FILE_2_matched', "rb") as f:
            matched_channels = pickle.load(f)

        for i in range(len(d2)):
            for j in range(len(d2[0])):
                self.assertTrue(torch.all(torch.eq(matched_channels[i][j],self.data2[i][j])))


    def test_match_channels_blur(self):
        # Tests match_channels(..., mode=blur)
        _, d2 = load_files(self.file1, self.file2)

        match_channels(self.file1, self.file2, 'blur')
        with open('./TEST_FILE_2_matched', "rb") as f:
            matched_channels = pickle.load(f)

        for i in range(len(d2)):
            for j in range(len(d2[0])):
                self.assertTrue(torch.all(torch.eq(matched_channels[i][j],self.data2[i][j])))


    def test_match_channels_dilation(self):
        # Tests match_channels(..., mode=dilation)
        _, d2 = load_files(self.file1, self.file2)

        match_channels(self.file1, self.file2, 'dilation')
        with open('./TEST_FILE_2_matched', "rb") as f:
            matched_channels = pickle.load(f)

        for i in range(len(d2)):
            for j in range(len(d2[0])):
                self.assertTrue(torch.all(torch.eq(matched_channels[i][j],self.data2[i][j])))


    def test_match_channels_pooling(self):
        # Tests match_channels(..., mode=pooling)
        pass


if __name__ == '__main__':
    unittest.main()
