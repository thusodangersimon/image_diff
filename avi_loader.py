from skimage.io import ImageCollection

class AVILoader(object):

    def __init__(self, video_file):
        self.video_file = video_file
    
    def __call__(self, frame):
        return video_read(self.video_file, frame)

    def get_frames(self, start, stop, step=1):
        '''Returns generator, get frames in range(start,stop, step)'''
        self.ic = ImageCollection(range(start, stop, step), load_func=self)
        for image in self.ic:
            yield image

    def get_frame(self, frame):
        '''Get a single frame'''
        return ImageCollection([frame], load_func=self)[0]
