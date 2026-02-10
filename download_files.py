'''
Author: Wenshan Wang
Date: 2024-09-06

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os
# import sys

from colorama import Fore, Style

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from os.path import isfile, join

# Local imports.
from os.path import isdir, isfile, join
import argparse

def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)

def print_warn(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)

def print_highlight(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)

class AirLabDownloader(object):
    def __init__(self, bucket_name = 'tartandrive') -> None:

        endpoint_url = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
        
        self.client = boto3.client("s3", endpoint_url=endpoint_url, config=Config(signature_version=UNSIGNED))
        self.bucket_name = bucket_name

    def download(self, filelist, output_dir):
        success_source_files, success_target_files = [], []
        for source_file_name in filelist:
            target_file_name = join(output_dir, source_file_name)
            # Create target directory if it does not exist
            import os
            target_dir = os.path.dirname(target_file_name)
            if not os.path.exists(target_dir):
                # make recursive directory
                os.makedirs(target_dir)
            # if isfile(target_file_name):
            #     print_error('Error: Target file {} already exists..'.format(target_file_name))
            #     continue
            #     # return False, success_source_files, success_target_files

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            try:
                resp = self.client.get_object(Bucket=self.bucket_name, Key=source_file_name)

                with open(target_file_name, "wb") as f:
                    for chunk in resp["Body"].iter_chunks(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            
            except Exception as e:
                print(f"Error: Failed to download {source_file_name} due to {e}.")
                continue

            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
            success_source_files.append(source_file_name)
            success_target_files.append(target_file_name)
        
        if len(success_target_files) == len(filelist):
            return True, success_source_files, success_target_files
        else:
            return False, success_source_files, success_target_files


class TartandriveDownloader():
    def __init__(self, ):
        super().__init__()

        self.downloader = AirLabDownloader()


    def unzip_files(self, zipfilelist, target_folder):
        print_warn('Note unzipping will overwrite existing files ...')
        for zipfile in zipfilelist:
            if not isfile(zipfile) or (not zipfile.endswith('.zip')):
                print_error("The zip file is missing {}".format(zipfile))
                return False
            print('  Unzipping {} ...'.format(zipfile))
            cmd = 'unzip -q -o ' + zipfile + ' -d ' + target_folder
            os.system(cmd)
        print_highlight("Unzipping Completed! ")
            
    def download(self, target_path, unzip = False, **kwargs):
        """
        """
        with open('azfiles.txt', 'r') as f:
            lines = f.readlines()

        zipfilelist = [ll.strip() for ll in lines] 

        suc, targetfilelist = self.downloader.download(zipfilelist, target_path)
        if suc:
            print_highlight("Download completed! Enjoy using Tartandrive!")

        if unzip:
            self.unzip_files(targetfilelist)

        return True

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--download-dir', default='./',
                        help='root directory for downloaded files')

    args = parser.parse_args()

    downloader = TartandriveDownloader()
    downloader.download(args.download_dir)
