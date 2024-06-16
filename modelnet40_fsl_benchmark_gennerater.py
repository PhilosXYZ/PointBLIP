import numpy as np
import os
import glob
import h5py
import ctypes
from utils.render_util import rende_xml

reflect = 3
light = 6
samples_per_pixel = 256
white_ball = True
resolution = 224


xml_head = \
    """
    <scene version="0.6.0">

        <integrator type="path">
            <integer name="max_depth" value="{}"/>
        </integrator>    



    """.format(reflect)

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="{}"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail_0 = \
    """    
        <sensor type="perspective">
            <transform name="to_world">
                <lookat origin="2.5,2.5,2.5" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="independent">
                <integer name="sample_count" value="16"/>
            </sampler>

            <film type="hdrfilm">
                <integer name="width" value="{}"/>
                <integer name="height" value="{}"/>
            </film>
        </sensor>    

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="{}"/>
            </emitter>
        </shape>               

    </scene>
    """.format(resolution, resolution, light)

xml_tail_1 = \
    """    
        <sensor type="perspective">
            <transform name="to_world">
                <lookat origin="2.5,-2.5,2.5" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="independent">
                <integer name="sample_count" value="16"/>
            </sampler>

            <film type="hdrfilm">
                <integer name="width" value="{}"/>
                <integer name="height" value="{}"/>
            </film>
        </sensor>    

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="{}"/>
            </emitter>
        </shape>               

    </scene>
    """.format(resolution, resolution, light)

xml_tail_2 = \
    """    
        <sensor type="perspective">
            <transform name="to_world">
                <lookat origin="-2.5,-2.5,2.5" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="independent">
                <integer name="sample_count" value="16"/>
            </sampler>

            <film type="hdrfilm">
                <integer name="width" value="{}"/>
                <integer name="height" value="{}"/>
            </film>
        </sensor>    

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="4,-4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="{}"/>
            </emitter>
        </shape>               

    </scene>
    """.format(resolution, resolution, light)

xml_tail_3 = \
    """    
        <sensor type="perspective">
            <transform name="to_world">
                <lookat origin="-2.5,2.5,2.5" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="independent">
                <integer name="sample_count" value="16"/>
            </sampler>

            <film type="hdrfilm">
                <integer name="width" value="{}"/>
                <integer name="height" value="{}"/>
            </film>
        </sensor>    

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,-4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="{}"/>
            </emitter>
        </shape>               

    </scene>
    """.format(resolution, resolution, light)

xml_tail = {
    '0': xml_tail_0,
    '1': xml_tail_1,
    '2': xml_tail_2,
    '3': xml_tail_3,

}


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


from tqdm import tqdm


if __name__ == '__main__':

    hdf5_file = "../../data/ModelNet_FSL/Modelnet40_FS_test.h5"
    f = h5py.File(hdf5_file, 'r')
    data = np.array(f['data'][:])
    label = np.array(f['label'][:])


    num_points = 2048
    save_path = "../../data/ModelNet_FSL/rendered_images"
    xml_path = os.path.join(save_path, 'xml')
    image_path = os.path.join(save_path, 'image')
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)


    for item in tqdm(range(data.shape[0])):

        pcl = data[item]
        pcl = standardize_bbox(pcl, num_points)
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.025

        min_dis_set = np.min(
            np.sqrt(np.sum((np.expand_dims(pcl, 1) - np.expand_dims(pcl, 0)) ** 2, axis=-1, keepdims=False))
            + np.eye(num_points) * 1000, axis=-1, keepdims=False)
        radius = min_dis_set.mean()

        xml_segments = [xml_head]

        for i in range(pcl.shape[0]):
            color = [0.8, 0.8, 0.8]
            xml_segments.append(xml_ball_segment.format(radius, pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))

        for i in xml_tail.keys():
            xml_per_view = xml_segments.copy()
            xml_per_view.append(xml_tail[i])
            xml_content = str.join('', xml_per_view)

            if os.path.exists(os.path.join(xml_path,'{}_{}.xml'.format(str(item), str(i)))):
                os.remove(os.path.join(xml_path,'{}_{}.xml'.format(str(item), str(i))))
            with open(os.path.join(xml_path,'{}_{}.xml'.format(str(item), str(i))), 'w') as f:
                f.write(xml_content)

            rende_xml(os.path.join(xml_path, '{}_{}.xml'.format(str(item), str(i))),
                      samples_per_pixel,
                      os.path.join(image_path, '{}_{}.png'.format(str(item), str(i))))