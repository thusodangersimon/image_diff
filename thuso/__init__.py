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

    def _get_inital_master(self, start, samples):
        '''samples previous frames from the video to get a inital master frame
        Start is the frame number to start, and samples is how many frames to
        sample'''
        # Get empty frame
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
        sucess, inital_frame = self.cap.read()
        while not sucess:
            sucess, inital_frame = self.cap.read()
        inital_frame = np.float32(inital_frame)
        # get random samples
        sam_array = np.arange(start, start+samples)
        count = 0.
        for frame_no in sam_array:
            #self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            # get frame
            sucess, frame = self.cap.read()
            if not sucess:
                continue
            cv2.accumulateWeighted(np.float32(frame), inital_frame, 0.020)
            count += 1

        return inital_frame

    def parallel_moving_ave(self, frame_array, n_sample=120):
        '''Does same as moving_ave_chi but in parrallel'''
        # Check for master image
        if not 'moving_ave_frame' in vars(self):
            # Check if at zero
            if np.min(frame_array) - n_sample < 0:
                start = 0
            else:
                start = np.min(frame_array) - n_sample
            self.moving_ave_frame = self._get_inital_master(start, n_sample)

        self.frame_no = frame_array
        self.frame_chi = []
        self.frame_time = []
        # start chi square
        x,y,z = self.moving_ave_frame.shape
        ddof = float(x * y)
        for frame_no in frame_array:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                # remove frame_no
                self.frame_no.pop(self.frame_no.index(frame_no))
                continue
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Cal chi
            frame = np.float32(frame)
            # add 1 so chi square can't be nan or inf
            self.frame_chi.append(chisquare(frame+1,
                                            self.moving_ave_frame+1).sum()
                                                                     / ddof)              
            # Cal new ave frame
            cv2.accumulateWeighted(frame, self.moving_ave_frame, 0.020)

        return self.frame_no, self.frame_time, self.frame_chi
        
    
    def moving_ave_chi(self, max_frames=-1, fps=1, plot=False):
        '''Uses moving average for master frame and calculates fps frames per second'''
        if max_frames < 1:
            max_frames = self.frames
        # set number of steps to skip
        stride = int(round(self.f_rate / float(fps)))
        # get inital aveage frame
        self.moving_ave_frame = self._get_inital_master(0, 120)
       
        # out puts
        self.frame_no = []
        self.frame_chi = []
        self.frame_time = []
        # start chi square
        x,y,z = self.moving_ave_frame.shape
        ddof = float(x * y)
        for frame_no in xrange(0, max_frames, stride):
            # Get chi square
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                continue
            self.frame_no.append(frame_no)
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Cal chi
            self.frame_chi.append(chisquare(frame, self.moving_ave_frame).sum()/ddof)
            # Cal new ave frame
            cv2.accumulateWeighted(frame, self.moving_ave_frame, 0.020)
            # Show moving Ave Frame and chi plot
            if plot:
                cv2.imshow('Ave Frame',self.moving_ave_frame)
                #cv2.waitKey(100)

    def _msec2min(self, msec):
        '''Converts miliseconds to min'''
        return msec * 1.6666666666666667e-05
    
    def _msec2sec(self, msec):
        '''Converts mili-sec to sec'''
        return msec * 10**-3

    def plot(self, individual=True):
        print "I am about to start drawing those amazing figures you really want to see!"
        majorLocator   = MultipleLocator(5)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)

        frame_no, frame_chi = self.frame_no, self.frame_chi
        # in mili seconds
        frameTime = np.asarray(self.frame_time)
        redChi = frame_chi

        self.dpi = 600
        # time in min
        #ipdb.set_trace()
        minTime, maxTime = self._msec2min(min(frameTime)), self._msec2min(max(frameTime))
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
        
        plt.ylim([minChi, maxChi])
        plt.xlim([minTime, maxTime])
        plt.xlabel("Time (min)")
        plt.ylabel("Reduced Chi-squared")

        plt.gca().set_yscale('log')

        xpos, ypos = float(minTime), float(maxChi/1.5)
        plt.text(xpos, ypos, self.video_path)
        '''
        if not individual:
            if self.rank == 1:
                plt.plot(self._msec2min(frameTime),redChi, 'r-')
                plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
                plt.clf()
        else:
            for i in xrange(len(frameTime)):
                pos = float(i)
                plt.axvline(x=pos)      
            plt.plot(self._msec2min(frameTime), redChi, 'k-')
            plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
            plt.show()
            plt.clf()
            '''
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
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        out = cv2.VideoWriter(self.video_path + '.mp4',fourcc, 50, (self.width,self.height))

        #if self.rank == 0:
        print "Making a movie, I am rank: ", self.rank
        #self.images = []
        #x_offset=y_offset= 0
        #self.comm.Barrier()
        #if self.rank == 0:
        for i in range(len(self.frame_no)):
        #for i_base in range(0, len(self.frame_no)-1, self.size):
            #i = i_base + self.rank
            if i < len(self.frame_no):
            #print self.frame_chi[j]
                if i == len(self.frame_no) and self.frame_chi[i] > 3.:
                    print "Oh be a fine girl kiss me now"
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] > 3.:
                    print "it's true", self.frame_no[i], self.rank, len(self.frame_no)
                #self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(self.frame_no[i]))
                #success, frame = self.cap.read()
                #if not success:
                #    continue
                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                        #self.comm.Barrier()
                        #self.images.append(j)
                    #self.frame = np.zeros(shape=(self.width,self.height))
                    #for k in range(50):
                        #out.write(self.frame)
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] < 3.:
                    print "not true", self.frame_no[i], self.rank, len(self.frame_no)
                #self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(self.frame_no[i]))
                #success, frame = self.cap.read()
                #if not success:
                #    continue
                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        #self.images.append(j)
                        out.write(frame)
                        #self.comm.Barrier()
                    #self.frame = np.zeros(shape=(self.width,self.height))
                    #for k in range(50):
                        #out.write(self.frame)

        #self.comm.Barrier()

        '''
        success, frame = self.cap.read()
        for i in range(len(self.images)):
        #for i in range(0, len(self.images)-1, self.size):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(self.images[i]))
            print self.rank, i, self.images[i]
            #self.comm.Barrier()
            self.new_images = frame
            out.write(self.new_images)
        '''
        '''
        for i in xrange(0, self.frames, 1):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
            success, frame = self.cap.read()
            if not success:
                continue

            images = frame
                #if self.frame_chi[i] > self.minChi:
            #print self.frame_chi[i]
                    #images = cv2.imread(files[i])
                    #s_img = cv2.imread(output[i])
                    #s_img = cv2.resize(s_img, (0,0), fx=0.5, fy=0.5) 
                    #images[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            out.write(images)
            '''
    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

    def mpi_work_split(self, max_frames=-1, size=None, fps=1, start=0):
        '''Splits work up from worker size'''
        if not size is None and size > 1:
            self.size = size
        if max_frames < 1:
            max_frames = self.frames
        
        stride = int(round(self.f_rate / float(fps)))
        # Make split array
        split = np.linspace(start, max_frames, self.size+1).round()
        frame_min, frame_max = int(split[self.rank]),int(split[self.rank + 1])
        return range(frame_min, frame_max, stride)
        

    def aggrigate_to_root(self, frame_no, frame_time, frame_chi):
        '''Send and organizes data at root'''
        if self.rank == 0:
            reciv =[(frame_no, frame_time, frame_chi)]
            for source in xrange(1,self.size):
                reciv.append(self.comm.recv(source=source))
                print 'Received from process %i'%source
            self.frame_no, self.frame_time, self.frame_chi = np.hstack(reciv)

        else:
            self.comm.send((frame_no, frame_time, frame_chi), dest=0)
            
    def save_result(self, filename=None):
        '''Saves results to a .csv file  with 3 colmns [frame_num,frame_time
        frame_chi].
        If no files name given with be video_name.csv'''
        if filename is None:
            outfile = os.path.splitext(self.video_path)[0] + '.csv'
        else:
            outfile = os.path.splitext(filename)[0] + '.csv'

        if not hasattr(self, 'frame_no'):
            raise AttributeError('Must get reduiced chi squared first')
        
        out_array = np.vstack((self.frame_no, self.frame_time,
                               self.frame_chi)).T
        np.savetxt(outfile, out_array, delimiter='\t',
                   header='frame_number, frame_time, frame_chi')
        
    
def chisquare(f_obs, f_exp, axis=0):
    '''calculates chi squared test correctly for 3D arrays'''
    return (f_obs - f_exp)**2 / f_exp



if __name__ == '__main__':
    import cPickle as pik
    # Test multiprocessing
    #frames = 10000
    video = Video_class('00016.MTS')
    # Single process
    '''if video.rank == 0:
        t_single = mpi.Wtime()
        video.moving_ave_chi()
        t_single -= mpi.Wtime()
        del video.moving_ave_frame'''
    video.comm.barrier()
    # Multiprocess
    t_multi = mpi.Wtime()
    work_array = video.mpi_work_split()
    frame_no, frame_time, frame_chi = video.parallel_moving_ave(work_array)
    video.aggrigate_to_root(frame_no, frame_time, frame_chi)
    video.comm.barrier()
    video.save_result()
    t_multi -= mpi.Wtime()
    if video.rank == 0:
        # Save
        pik.dump((video.frame_no, video.frame_time, video.frame_chi),open('t','w'),2)
        #print 'Single time is %2.0f and Multi time is %2.0f for speed up of %2.1f times'%(abs(t_single),abs(t_multi), t_multi/t_single)
        #video.plot()
        #video.comm.barrier()
        video.make_movie()
