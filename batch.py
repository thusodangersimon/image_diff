#/usr/bin/python2

'''
time mpiexec -n 8 python batch.py movie_filename
'''

import numpy as np
import scipy as sp
from scipy import misc
from scipy import stats
import mpi4py.MPI as mpi
import pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime as dt
import os
import sys
import cv2
import glob
import time
#import ipdb


class Video_class(object):

    def __init__(self, video_path):
        self.comm = mpi.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #video info
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.f_rate = int(self.cap.get(cv2.cv.CV_CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.f_sec = 1 / self.f_rate
        self.width = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        #h is the step size. Essentially you want one frame at a time!
        self.h = 1

        #just a random stating point
        self.a = 0

        #local_i is the number of frames each processor will use
        self.local_i = self.frames / ( self.size )

        #calculate interval that each process handles
        self.local_startframe = self.a + ( self.rank ) * self.local_i * self.h
        self.local_endframe = self.local_startframe + self.local_i * self.h - 1

        #print self.local_startframe , self.local_endframe , self.rank

    def make_master_im(self, max_frames=-1, step=-1):
        '''makes master in memory'''
        if max_frames < 1 and step < 1:
            max_frames = self.frames
            step = self.f_rate
        elif max_frames < 1 and step >= 1:
            max_frames = self.frames
        elif max_frames >= 1 and step < 1:
            step = self.local_i / (max_frames / self.size)

        print step, max_frames

        #print self.local_i, step, max_frames / self.size
        #initalize master image
        out = []
        _, self.master_image = self.cap.read()
        sample = np.unravel_index(np.random.randint(0, len(self.master_image.ravel())),self.master_image.shape)
        out.append(self.master_image[sample])
        self.master_image = np.asarray(self.master_image, dtype=float)
        #for image_no in xrange(1, max_frames):
        for image_no in xrange(self.local_startframe, self.local_endframe, step):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, image_no)
            print image_no, step, self.local_startframe, self.local_endframe, self.frames
            #per_done = self.cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)*100.
            #per_done = self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)100.
            #if per_done % 10 == 0:
            #print '%2.0f Percent Done'%per_done, image_no
            _, temp_image = self.cap.read()
            out.append(temp_image)
            temp_image = np.asarray(temp_image, dtype=float)
            self.master_image += temp_image
        self.master_image /= float(max_frames)
        #self.master_image = self.master_image.clip(0, 255)
        #self.master_image = np.uint8(self.master_image, dtype=np.uint8)
        return out
    
    def image_chi(self, max_frames=-1, step=-1):
        '''
        Calculate chi squared on the images
        '''
        if max_frames < 1 and step < 1:
            max_frames = self.frames
            step = self.f_rate
        elif max_frames < 1 and step >= 1:
            max_frames = self.frames
        elif max_frames >= 1 and step < 1:
            step = self.local_i / (max_frames / self.size)
        
        self.frame_no = []
        self.frame_chi = []
        print "About to start doing Chi-squared"
        x, y, z = self.master_image.shape
        ddof = x*y
        #for i_base in xrange(0, max_frames):
        for i_base in xrange(self.local_startframe, self.local_endframe, step):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i_base)
            print i_base, step, self.local_startframe, self.local_endframe, self.frames 
            #per_done = self.cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)*100.
            #if per_done % 10 == 0:
            #print '%2.0f Percent Done'%per_done, i_base
            #print self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            _, images = self.cap.read()
            # Scale images so there are non nans or inf from chi squared test
            chi = chisquare(np.float32(images)+1, self.master_image+1)
            reduced = np.sum(chi) / ( ddof )
            self.frame_no.append(float(i_base))
            self.frame_chi.append(float(reduced))
            
        self.frame_no = np.asarray(self.frame_no)
        self.frame_chi = np.asarray(self.frame_chi)
        self.comm.Barrier()

        print "About to start zipping..."
        with open(self.video_path + '.dat', 'a') as outfile:
            print "About to start zipping... I am rank ", self.rank
            zipped = zip(self.frame_no, self.frame_chi)
            np.savetxt(outfile, zipped)
        self.comm.Barrier()
    #THIS FILE OUTPUT WILL BE USED FOR POSTPROCESSING IF NEEDS BE

    def plot(self, individual=True):
        print "I am about to start drawing those amazing figures you really want to see!"
        majorLocator   = MultipleLocator(5)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)

        frame_no, frame_chi = self.frame_no, self.frame_chi
        frameTime = frame_no / 60.
        redChi = frame_chi

        self.dpi = 600

        minTime, maxTime = min(frameTime), max(frameTime)
        minChi, maxChi = min(redChi), max(redChi)
        averageChi = float(sum(redChi) / len(redChi))

        #plt.axhline(y=3*minChi)

        ax = plt.subplot(1, 1, 1)
        #plt.figure()
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
    
        plt.tick_params(axis='both', which='major', length=10)
        plt.tick_params(axis='both', which='minor', length=5)
        
        #plt.ylim([minChi, maxChi])
        #plt.xlim([minTime, maxTime])
        #plt.xlabel("Time (min)")
        #plt.ylabel("Reduced Chi-squared")

        plt.gca().set_yscale('log')

        xpos, ypos = float(minTime), float(maxChi/1.5)
        plt.text(xpos, ypos, self.video_path)

        if individual == False:
            if self.rank == 1:
                plt.plot(frameTime,redChi, 'r-')
                plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
                plt.clf()
        elif individual == True:
            for i in xrange(len(frameTime)):
                pos = float(i / 60.)
                plt.axvline(x=pos)      
                plt.plot(frameTime, redChi, 'k-')
                plt.savefig(output, dpi=self.dpi)
                plt.clf()
        #plt.savefig(temp_folder + "reduced.eps")
        #plt.savefig(rootfile + ".eps")
        #ipdb.set_trace()
        #plt.show()
        '''
        for i in xrange(len(frameTime)):
            output = (temp_folder + "image%.7d.jpg" % i)

            ax = plt.subplot(1, 1, 1)

            plt.axhline(y=3*minChi)

            #plt.figure()
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_major_formatter(majorFormatter)
            ax.xaxis.set_minor_locator(minorLocator)

            plt.tick_params(axis='both', which='major', length=10)
            plt.tick_params(axis='both', which='minor', length=5)
        
            plt.ylim([minChi, maxChi])
            plt.xlim([minTime, maxTime])
            plt.xlabel("Time (min)")
            plt.ylabel("Reduced Chi-squared")

            plt.gca().set_yscale('log')
    
            xpos, ypos = float(minTime), float(maxChi/1.5)
            plt.text(xpos, ypos, rootfile)

            pos = float(i / 60.)
            #print output, pos
            plt.axvline(x=pos)
            plt.plot(frameTime, redChi, 'k-')
            plt.savefig(output)
            plt.clf()
            '''
    def make_movie(self):
        '''
    TO CREATE MOVIE
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    '''

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(self.video_path + '.avi',fourcc, 1, (self.width,self.height))
        if self.rank == 0:
            print "Making a movie"
            x_offset=y_offset= 0
            for i in xrange(0, len(self.frame_no), 1):
                #if self.frame_chi[i] > self.minChi:
                print self.frame_chi[i]
                    #images = cv2.imread(files[i])
                    #s_img = cv2.imread(output[i])
                    #s_img = cv2.resize(s_img, (0,0), fx=0.5, fy=0.5) 
                    #images[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                    #out.write()

    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

def chisquare(f_obs, f_exp, axis=0):
    '''calculates chi squared test correctly for 3D arrays'''
    return (f_obs - f_exp)**2 / f_exp

if __name__ == '__main__':

    # Set the Number of frames wished
    frames = 30
    video = Video_class('00016.MTS')
    freq = video.make_master_im(frames)
    #plt.hist(freq, 5)
    #plt.show()
    video.image_chi()
    #video.plot()
    #video.make_movie()
