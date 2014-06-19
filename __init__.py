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
import ipdb


class Video_class(object):

    def __init__(self, video_path):
        self.comm = mpi.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #video info
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.f_rate = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frames = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.f_sec = 1 / self.f_rate
        self.width = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    def video_to_image(self, step_size=1, frame_start=0):
        '''Takes the videos and outputs frames into images'''
        #split files
        rootfile, ext = sys.argv[1].split(".")

        #CHECK IF FOLDER EXISTS
        temp_folder = "temp_folder2/"
        master_file = os.path.join(temp_folder, "all_averaged.jpg")

        if rank == 0:
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)

        #h is the step size. Essentially you want one frame at a time!
        h = 1
    
        #just a random stating point
        a = 0

        #local_i is the number of frames each processor will use
        local_i = frames / ( size )

        #calculate interval that each process handles
        local_startframe = a + ( rank ) * local_i * h
        local_endframe = local_startframe + local_i * h - 1

        #print local_startframe , local_endframe , frames
    
        for i in xrange(local_startframe, local_endframe, int(f_rate)):
            #print i
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
            success,image = cap.read()
            if i < frames:
                cv2.imwrite(temp_folder + "frame%.7d.jpg" % i, image)
        comm.Barrier()

    def make_master_im(self, max_frames=-1):
        '''makes master in memory'''
        if max_frames < 1:
            max_frames = self.frames
        #initalize master image
        out = []
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        _, self.master_image = self.cap.read()
        sample = np.unravel_index(np.random.randint(0, len(self.master_image.ravel())),self.master_image.shape)
        out.append(self.master_image[sample])
        self.master_image = np.asarray(self.master_image, dtype=float)
        for image_no in xrange(1, max_frames):
            per_done = self.cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)*100.
            #if per_done % 10 == 0:
            print '%2.0f Percent Done'%per_done, image_no
            _, temp_image = self.cap.read()
            out.append(temp_image[sample])
            temp_image = np.asarray(temp_image, dtype=float)
            self.master_image += temp_image
        self.master_image /= float(max_frames)
        #self.master_image = self.master_image.clip(0, 255)
        #self.master_image = np.uint8(self.master_image, dtype=np.uint8)
        return out
    
    def image_chi(self, max_frames=-1):
        '''
        Calculate chi squared on the images
        '''
        if max_frames < 1:
            max_frames = self.frames
        
        self.frame_no = []
        self.frame_chi = []
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        print "About to start doing Chi-squared"
        x, y, z = self.master_image.shape
        ddof = x*y
        for i_base in xrange(0, max_frames):
            per_done = self.cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)*100.
            #if per_done % 10 == 0:
            print '%2.0f Percent Done'%per_done, i_base
            print self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            _, images = self.cap.read()
            # Scale images so there are non nans or inf from chi squared test
            chi = chisquare(np.float32(images)+1, self.master_image+1)
            reduced = np.sum(chi) / ( ddof )
            self.frame_no.append(float(i_base))
            self.frame_chi.append(float(reduced))
            
        self.frame_no = np.asarray(self.frame_no)
        self.frame_chi = np.asarray(self.frame_chi)
        '''#print "About to start zipping..."
        with open(temp_folder + 'results.dat', 'a') as outfile:
        print "About to start zipping... I am rank ", rank
        zipped = zip(frame_no, frame_chi)
        np.savetxt(outfile, zipped)
        comm.Barrier()'''
    #THIS FILE OUTPUT WILL BE USED FOR POSTPROCESSING IF NEEDS BE

    def plot(self):
        print "I am about to start drawing those amazing figures you really want to see!"
        majorLocator   = MultipleLocator(5)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)

        
        frame_no, frame_chi = self.frame_no, self.frame_chi
        frameTime = frame_no / 60.
        redChi = frame_chi
        
        minTime, maxTime = min(frameTime), max(frameTime)
        minChi, maxChi = min(redChi), max(redChi)
        averageChi = float(sum(redChi) / len(redChi))

        plt.axhline(y=3*minChi)

        ax = plt.subplot(1, 1, 1)
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
        plt.text(xpos, ypos, self.video_path)

        plt.plot(frameTime,redChi, 'r-')
        #plt.savefig(temp_folder + "reduced.eps")
        #plt.savefig(rootfile + ".eps")
        #ipdb.set_trace()
        plt.show()
        plt.savefig(self.video_path + ".jpg")
    
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

    def make_moive(self):
        '''
    TO CREATE MOVIE
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    '''

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(rootfile + '.avi',fourcc, 1, (width,height))

        print "Making a movie"
        x_offset=y_offset= 0
        files = glob.glob(temp_folder + "/frame*.jpg")
        files.sort()
        del files[-1]
        output = glob.glob(temp_folder + "image*.jpg")
        output.sort()
        for i in xrange(0, len(files), 1):
            if frame_chi[i] > 3*minChi:
                images = cv2.imread(files[i])
                s_img = cv2.imread(output[i])
                s_img = cv2.resize(s_img, (0,0), fx=0.5, fy=0.5) 
                images[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                out.write(images)

    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

def chisquare(f_obs, f_exp, axis=0):
    '''calculates chi squared test correctly for 3D arrays'''
    return (f_obs - f_exp)**2 / f_exp

if __name__ == '__main__':

    video = Video_class('00016.MTS')
    freq = video.make_master_im(30)
    #plt.hist(freq, 5)
    #plt.show()
    video.image_chi(30)
    video.plot()
