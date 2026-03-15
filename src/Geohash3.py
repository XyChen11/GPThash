import numpy as np
import torch
from math import log10

__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__decodemap = { }
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i
del i

def decode_geohash(binary_geohash):
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    hei_interval = [0.0, 500.0]
    
    flag = 0
    
    for bit in binary_geohash:
        if flag == 0:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if bit == '1':
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        elif flag == 1:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if bit == '1':
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        elif flag == 2:
            mid = (hei_interval[0] + hei_interval[1]) / 2
            if bit == '1':
                hei_interval[0] = mid
            else:
                hei_interval[1] = mid

        if flag < 3:
                flag += 1
        else:
                flag = 0
    
    latitude = (lat_interval[0] + lat_interval[1]) / 2
    longitude = (lon_interval[0] + lon_interval[1]) / 2
    height = (hei_interval[0] + hei_interval[1]) / 2
    return latitude, longitude, height

def decode3_exactly(geohash3):
    """
    Decode the geohash to its exact values, including the error
    margins of the result.  Returns six float values: latitude,
    longitude, height, the plus/minus error for latitude (as a positive
    number), the plus/minus error for longitude (as a positive
    number) and the plus/minus error for height (as a positive
    number).
    """
    lat_interval, lon_interval, hei_interval = (-90.0, 90.0), (-180.0, 180.0), (0.0, 500.0)
    lat_err, lon_err, hei_err = 90.0, 180.0, 250.0
    flag = 0
    for c in geohash3:
        if c not in __decodemap:
            continue  # 跳过所有无效字符 包括大写
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if flag == 0: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            if flag == 1:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            if flag == 2:
                hei_err /= 2
                if cd & mask:
                    hei_interval = ((hei_interval[0]+hei_interval[1])/2, hei_interval[1])
                else:
                    hei_interval = (hei_interval[0], (hei_interval[0]+hei_interval[1])/2)

            if flag < 3:
                flag += 1
            else:
                flag = 0

    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    hei = (hei_interval[0] + hei_interval[1]) / 2
    return lat, lon, hei, lat_err, lon_err, hei_err

def decode3(geohash):
    """
    Decode geohash, returning two strings with latitude and longitude
    containing only relevant digits and with trailing zeroes removed.
    """
    lat, lon, hei, lat_err, lon_err, hei_err = decode3_exactly(geohash)
    # Format to the number of decimals that are known
    lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
    lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
    heis = "%.*f" % (max(1, int(round(-log10(hei_err)))) - 1, hei)
    if '.' in lats: lats = lats.rstrip('0')
    if '.' in lons: lons = lons.rstrip('0')
    if '.' in heis: heis = lons.rstrip('0')
    return lats, lons, heis

def encode3(latitude, longitude, height, precision=12):
    """
    Encode a position given in float arguments latitude, longitude and height to
    a geohash3 which will have the character count precision.
    """
    lat_interval, lon_interval, height_interval = (-90.0, 90.0), (-180.0, 180.0), (0.0, 500.0)
    geohash3 = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    flag = 0
    while len(geohash3) < precision:
        if flag == 0:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        if flag == 1:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        if flag == 2:
            mid = (height_interval[0] + height_interval[1]) / 2
            if height > mid:
                ch |= bits[bit]
                height_interval = (mid, height_interval[1])
            else:
                height_interval = (height_interval[0], mid)

        if flag < 3:
            flag += 1
        else: 
            flag = 0

        if bit < 4:
            bit += 1
        else:
            geohash3 += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash3)
