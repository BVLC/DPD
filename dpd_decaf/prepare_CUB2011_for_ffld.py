import os
import shutil
import glob

def parse_train_test(trainTestLine):
    [imgID, trainOrTest] = trainTestLine.split()

    if(trainOrTest == '1'):
        return 'train'
    elif(trainOrTest == '0'):
        return 'test'
    else:
        return 'not train or test, wtf?'

#one file per image ID:
# input  file format: xmin, xmax, width, height
# output file format: xmin, ymin, xmax, ymax
def prepare_ffld_bird_labels(outTxtLocation, birdLocation, numImgs):

    boxFile = open(birdLocation + '/bounding_boxes.txt', 'r')
    trainTestFile = open(birdLocation + '/train_test_split.txt', 'r')

    for imgID in xrange(1, numImgs+1):
        boxLine = boxFile.readline().rstrip('\n') #next line
        [myImgID, xmin, ymin, width, height] = boxLine.split()

        trainTestLine = trainTestFile.readline().rstrip('\n')
        trainTestStr = parse_train_test(trainTestLine)

        outFile = open(outTxtLocation + '/' + trainTestStr + '/' + "%05d.txt"%imgID, 'w')
        outFile.write('%f, %f, %f, %f \n' % (float(xmin), float(ymin), float(xmin)+float(width), float(ymin)+float(height))) # xmin, ymin, xmax, ymax
        outFile.close()

# e.g. CUB_200_2011/images/013.Bobolink/Bobolink_0002_11085.jpg -> dpd_scratch/.../test/00699.jpg
def prepare_ffld_bird_images(outImgLocation, birdLocation, numImgs):
    imgIdxFile = open(birdLocation + '/images.txt', 'r')
    trainTestFile = open(birdLocation + '/train_test_split.txt', 'r')
    
    for imgID in xrange(1, numImgs+1):
        imgIdxLine = imgIdxFile.readline().rstrip('\n') #next line
        trainTestLine = trainTestFile.readline().rstrip('\n')
        trainTestStr = parse_train_test(trainTestLine)

        [myImgID, imgFname] = imgIdxLine.split()
        assert(imgID == int(myImgID))

        oldImgFname = birdLocation + '/images/' + imgFname
        newImgFname = outImgLocation + '/' + trainTestStr +  '/' + "%05d.jpg"%imgID
        shutil.copy(oldImgFname, newImgFname)

#assume all files have the same extension (e.g. jpg or txt)
# move [00004.jpg, 00009.jpg, ....] -> [00001.jpg, 00002.jpg, ...]
# this is fragile, use with caution.
def reindex_from_1(directory):
    files = os.listdir(directory)
    files = sorted(files)
    fileIdx = 1

    for f in files:
        old_Fname = directory + '/' + f
        [basefile, ext] = os.path.splitext(f)
        new_Fname = directory + '/' + "%05d"%fileIdx + ext
        shutil.move(old_Fname, new_Fname)
        fileIdx = fileIdx + 1

#unlike ordinary os.remove(), this handles wildcards, e.g. ./blah*/*
def remove_with_wildcard(directory):
    for fl in glob.glob(directory):
        os.remove(fl)

birdLocation = './CUB_200_2011' #input
outTxtLocation = '/media/big_disk/dpd_scratch/bird_labels_for_ffld'
outImgLocation = '/media/big_disk/dpd_scratch/bird_images_for_ffld'
numImgs = 11788 #TODO: parse based on number of lines in train_test_split.txt 
#TODO: mkdir outTxtLocation, outImgLocation/train, outImgLocation/test

#remove any old data in our output directories
remove_with_wildcard(outTxtLocation + '/*/*') #outTestLocation/{train and test}/*txt
#remove_with_wildcard(outImgLocation + '/*/*') #outTestLocation/{train and test}/*jpg

#the crux -- copy data to flat directories and one bbox file per img
prepare_ffld_bird_labels(outTxtLocation, birdLocation, numImgs)
#prepare_ffld_bird_images(outImgLocation, birdLocation, numImgs) #bbox labels

reindex_from_1(outTxtLocation + '/train')
reindex_from_1(outTxtLocation + '/test')
#reindex_from_1(outImgLocation + '/train')
#reindex_from_1(outImgLocation + '/test')


