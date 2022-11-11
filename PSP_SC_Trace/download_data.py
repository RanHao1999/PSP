"""
@authr = RanHao

Aim:
This python script is for downloading data that is needed for analysis.
This script is supposed to be put in the main folder.
This script is written for provide downloading functions for SC2SolarSurface.py, it has no main function.

"""
import datetime
# import modules
import os
import requests
from bs4 import BeautifulSoup
import gzip

def mkd(path):
    folders = path.split('/')
    for i in range(1, len(folders)):
        if os.path.exists('/'.join(folders[0:i+1])) == False:
            os.mkdir('/'.join(folders[0:i+1]))


def download_file(url, target_folder):
    """
    download the url file to the target folder
    """

    def is_downloadable(url):
        """
        Does the url contain a downloadable resource.
        """
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            print('Data not downloadable')
            return False
        if 'html' in content_type.lower():
            print('Data not downloadable')
            return False
        return True

    if os.path.exists(target_folder) == False:
        mkd(target_folder)

    local_filename = target_folder + '/' + url.split('/')[-1]
    # make folder

    if os.path.exists(local_filename) == True:
        print('data already exists')
        return 0
    else:
        if is_downloadable(url) == True:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_filename
        else:
            print('There is no downloadable file linked to your url.')
            return 0

def un_gz(file_path):
    """
    unzip the .gz file.
    """
    f_name = file_path.split('/')[-1].replace('.gz', '')
    folders_lst = file_path.split('/')[:-1]
    folders_path = '/'.join(folders_lst)
    # unzip
    g_file = gzip.GzipFile(file_path)
    open(folders_path + '/' + f_name, 'wb+').write(g_file.read())
    g_file.close()
    return 0

def download_gong_adapt(time):
    """
    Functoin for downloading gong synoptic magnetogram (1h cadence) on a specific day.
    :param time: format: 'YYYY-mm-ddTHH(even number):00:00'. The temporal resolution for gong adapt is 2hr.
    :return: datas are saved in the 'data' folder
    """
    # download the gong adapt data
    target_folder = 'data/gong/adapt/' + time.split('T')[0].replace('-', '_') + '/' + time.split('T')[1].replace(':', '')
    date = time.split('T')[0]
    t_tag = time.split('T')[1]
    year, month, date = date.split('-')[0], date.split('-')[1], \
                        date.split('-')[2]
    hour, minute, second = t_tag.split(':')[0], t_tag.split(':')[1], \
                           t_tag.split(':')[2]
    time_tag = year + month + date + hour + minute

    url = 'https://gong.nso.edu/adapt/maps/gong/'+year
    r = requests.get(url, allow_redirects=True)
    soup = BeautifulSoup(r.text, 'html.parser')
    file_names = [node.get('href') for node in soup.find_all('a')
                  if node.get('href').endswith('fts.gz') and time_tag in node.get('href')]
    for file in file_names:
        download_file(url + '/' + file, target_folder)
#        un_gz(target_folder + '/' + file)
#        os.remove(target_folder + '/' + file)

    return 0
    # ================================================================

def download_psp(t_start, t_end):
    """
    Download a series of aia maps from jsoc website.
    :param date: format: 'YYYY-mm-dd'
    :return: datas are saved in the 'data' folder
    """
    # download the aia.fits
    def dates_between(t_start, t_end):
        date_list = []
        date_start = datetime.datetime.strptime(t_start, '%Y-%m-%dT%H:%M:%S')
        date_end = datetime.datetime.strptime(t_end, '%Y-%m-%dT%H:%M:%S') + datetime.timedelta(days=1)

        while date_start <= date_end:
            date_str = date_start.strftime('%Y-%m-%d')
            date_list.append(date_str)
            date_start += datetime.timedelta(days=1)
        return date_list

    target_folder = 'data/psp/' + t_start.split('T')[0].replace('-', '_')

    date_list = dates_between(t_start, t_end)

    for date in date_list:
        year, month, day = date.split('-')[0], date.split('-')[1], \
                            date.split('-')[2]
        url1 = 'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/' + year
        url2 = 'https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/' + year
        name1 = 'psp_swp_spi_sf00_l3_mom_' + date.replace('-', '') + '_v04.cdf'
        name2 = 'psp_swp_spc_l3i_' + date.replace('-', '') + '_v02.cdf'

        download_file(url1 + '/' + name1, target_folder)
        download_file(url2 + '/' + name2, target_folder)
    return 0

def download_aia_synoptic(CR_number):
    """
    Download AIA synoptic map from: https://sun.njit.edu/#/coronal_holes (IDSEAR web)
    :param CR_number: The carrington number of the AIA synoptic map
    :return: data in the folder
    """

    target_folder = 'data/aia'

    url = 'https://sun.njit.edu/coronal_holes/data/aia193_synmap_cr' + str(CR_number) + '.fits'
    download_file(url, target_folder)

    return 'aia193_synmap_cr' + str(CR_number) + '.fits'
