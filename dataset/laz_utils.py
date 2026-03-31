import io
import laspy
import numpy as np
import os
from time import time

def save_pcd_laz(path, pcd_numpy, res=0.1):
    """
    Save as .laz format, precision 0.1mm
    """
    # 1. Create Header (version 1.2, point format 3 supports GPS time and color)
    header = laspy.LasHeader(point_format=3, version="1.2")

    # 2. Key setting: Scale is res (default mm)
    res *= 0.001  # Convert to meters
    header.scales = np.array([res, res, res])

    # 3. Set Offset: To maximize compression efficiency, set offset to the minimum of current point cloud
    # This way stored integer values are smaller, differential encoding works better
    if pcd_numpy.shape[0] > 0:
        header.offsets = np.min(pcd_numpy[:, :3], axis=0)
    else:
        header.offsets = np.array([0, 0, 0])

    # 4. Create LasData
    las = laspy.LasData(header)

    # 5. Assign values (laspy will automatically apply offset and scale to convert to integers)
    las.x = pcd_numpy[:, 0]
    las.y = pcd_numpy[:, 1]
    las.z = pcd_numpy[:, 2]

    # Process intensity (if there is a 4th column)
    if pcd_numpy.shape[1] > 3:
        las.intensity = pcd_numpy[:, 3].astype(np.uint16)

    # 6. Write to file (extension is .laz will auto-compress)
    las.write(path)

def load_pcd_laz(path):
    """
    Read .laz and convert back to float32
    """
    if not os.path.exists(path):
        return None

    # t0 = time()
    with laspy.open(path) as f:
        las = f.read()
        # las.x/y/z automatically applied scale and offset, returns float64
        xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)

        # Process intensity
        try:
            intensity = las.intensity.astype(np.float32)
            pcd = np.column_stack((xyz, intensity))
        except:
            pcd = xyz
        # t1 = time()
        # print(f"Loaded LAZ {path} in {(t1 - t0) * 1000:.3f} milliseconds.")
        return np.ascontiguousarray(pcd)

def save_pcd_laz_to_bytes(pcd_numpy):
    """
    Save point cloud as .laz format binary stream (bytes)
    """
    # 1. Create Header (version 1.2, point format 3 supports GPS time and color)
    header = laspy.LasHeader(point_format=3, version="1.2")

    # 2. Key setting: Scale is 0.0001 (0.1mm)
    header.scales = np.array([0.0001, 0.0001, 0.0001])

    # 3. Set Offset: Maximize compression efficiency
    if pcd_numpy.shape[0] > 0:
        header.offsets = np.min(pcd_numpy[:, :3], axis=0)
    else:
        header.offsets = np.array([0, 0, 0])

    # 4. Create LasData
    las = laspy.LasData(header)

    # 5. Assign values
    las.x = pcd_numpy[:, 0]
    las.y = pcd_numpy[:, 1]
    las.z = pcd_numpy[:, 2]

    # Process intensity (if there is a 4th column)
    if pcd_numpy.shape[1] > 3:
        # Note: In LAZ standard intensity is usually uint16
        las.intensity = pcd_numpy[:, 3].astype(np.uint16)

    # 6. Write to memory stream
    with io.BytesIO() as out_stream:
        # Key changes:
        # 1. Pass stream object
        # 2. do_compress=True forces LAZ compression (since no filename extension to infer from)
        las.write(out_stream, do_compress=True)

        # Get binary data
        laz_bytes = out_stream.getvalue()

    return laz_bytes

def load_pcd_laz_bytes(laz_bytes):
    """
    Core processing logic: Parse binary stream to PCD numpy array
    """
    if not laz_bytes:
        return None

    # Wrap bytes as file-like object
    with io.BytesIO(laz_bytes) as stream:
        # laspy reads from stream
        with laspy.open(stream) as f:
            las = f.read()

            # Extract coordinates (automatically apply scale/offset)
            xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)

            # Extract intensity (if available)
            try:
                intensity = las.intensity.astype(np.float32)
                pcd = np.column_stack((xyz, intensity))
            except AttributeError:
                pcd = xyz

    return np.ascontiguousarray(pcd)
