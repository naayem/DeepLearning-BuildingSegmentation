import os
import shutil
import argparse
from glob import glob
import pandas as pd

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_segment_folder(df, streams_dir, segment_dir):
    id_segment = df['idsegment']
    # find the streams taken by panorana camera (the second item of streams)
    panorama_stream = df['streams'].split(',')[1]

    make_dir(f'{segment_dir}/{id_segment}')
    former_address = f'{streams_dir}/{panorama_stream}'
    frames = sorted(glob(f'{former_address}/*'))
    for frame in frames:
        frame_num = frame.split('/')[-1]
        # only use the direction 1 and direction 4 (left and right side of panorama camera)
        imgs = glob(frame+'/[14].jpg')
        for img in imgs:
            img_name = img.split('/')[-1]
            shutil.copy(
                img, f'{segment_dir}/{id_segment}/{id_segment}_{panorama_stream}_{frame_num}_{img_name}')

def main(args):
    df_segments_info = pd.read_csv(args.segments_info_path) # read the csv file
    test_segments = [item for item in args.id_segments.split(',')]
    df_test = df_segments_info[df_segments_info['idsegment'].isin(test_segments)]
    df_test.apply(lambda x: create_segment_folder(x, args.streams_dir, args.segment_image_path), axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    parser.add_argument('--segments_info_path', default='', type=str,
                        help="csv file to show the relation between segments and streams")
    parser.add_argument('--streams_dir', default='', type=str,
                        help="orginal building images are grouped by streams")
    parser.add_argument('--segment_image_path', default='', type=str,
                        help="select the direction 1 and 4 from the panorama camera and grouped by segments")
    parser.add_argument('--id_segments', default='16878,16888', type=str,
                        help="segments' id for testing(delimited list input)")
    args = parser.parse_args()

    main(args)
    print('============  Arguments infos ============ ')
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))