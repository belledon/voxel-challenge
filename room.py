#!/usr/bin/env python3


import time
from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np


scene = Scene(voxel_edges=0, exposure=10,
              voxel_grid_res = 64)

n = 32
hn = n // 2
height = 2 * n // 3 - hn
oheight = 1 * n // 6 - hn


scene.set_floor(-hn, (1.0, 1.0, 1.0))


@ti.kernel
def initialize_background():

    # left, right walls
    for i in range(-hn, height):
        for j in range(-hn, hn):
            # red (left wall)
            scene.set_voxel(vec3(-hn, i, j), 1, vec3(0.9, 0.3, 0.3))
            # green (right wall)
            scene.set_voxel(vec3(hn, i, j), 1, vec3(0.3, 0.9, 0.3))
            # back wall
            scene.set_voxel(vec3(j, i, -hn), 1, vec3(1, 1, 1))

    # ceiling and floor
    for i in range(-hn, hn):
        for j in range(-hn, hn):
            scene.set_voxel(vec3(i, height, j), 1, vec3(1, 1, 1))
            scene.set_voxel(vec3(i, -hn, j), 1, vec3(1, 1, 1))

    # add lights to the ceiling
    for i in range(-n // 8, n // 8):
        for j in range(-n // 8, n // 8):
            pos = vec3(i, height, j)
            # print(f'pos: {pos}')
            scene.set_voxel(pos, 2, vec3(1, 1, 1))


obstacles = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype = np.int32).repeat(2, 0).repeat(2, 1)

@ti.kernel
def initialize_obstacles(omap : ti.types.ndarray(dtype=ti.i32, ndim=2)):
    blue = vec3(0.3, 0.3, 0.9)

    for I in ti.grouped(omap):
        if omap[I]:
            x = I[0] - hn
            z = I[1] - hn
            for y in range(-hn, oheight):
                scene.set_voxel(vec3(x, y, z), 1, blue)


initialize_background()
initialize_obstacles(obstacles)
scene.finish()
