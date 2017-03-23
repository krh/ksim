# ksim
The little simulator that could.

ksim is a simulator for Intel Skylake GPUs. It grew out of two tools
I wrote a while back for [capturing](https://cgit.freedesktop.org/xorg/app/intel-gpu-tools/tree/tools/aubdump.c)
and [decoding](https://cgit.freedesktop.org/mesa/mesa/tree/src/intel/tools/aubinator.c)
the output of the Intel open source drivers.
Once you're capturing and decoding the command stream output by the driver it's a small step
to start interpreting the stream.  Initially, it was exciting to see it fetch vertices, but it
has now snowballed into a fairly competent and efficient software rasterizer. As of March 22nd, ksim runs
all major GL/Vulkan shader stages (vertex, hull, domain, geometry and fragment shaders) as well as
compute shaders. It JIT compiles the Intel GPU ISA to AVX2 code on-the-fly using its own IR and compiler.

ksim is far from complete and probably never will be. It was never intended to be useful but it's
been, and continue to be, a fun and educational project. That said, many half-baked parts of ksim
are falling into place and functionality I thought I'd never enable is now running. One day it might be
useful as a way to provide a simulated GPU for Qemu or, together with the Intel open source Vulkan driver,
a Vulkan software rasterizer.

## Building and running
ksim uses the meson build system so after cloning the repo building it boils down to:

```sh
$ mkdir build
$ meson . build --buildtype=release
$ ninja -C build
$ sudo ninja -C build install
```
This sets up a meson release build, which is going build a much faster ksim, but for debugging
you'll want --buildtype=debug. On Fedora, the `ninja` tool is called `ninja-build`.

Once installed, it's easy to launch any GL or Vulkan application under ksim:

````sh
$ ksim glxgears
````

will bring up the old gears running on ksim. Since ksim simulates the GPU from the kernel level down,
it can also run Vulkan applications. Some of the demos from@Sascha Willems
[vulkan demos](https://github.com/SaschaWillems/Vulkan) run and look mostly right.

The ksim executable is a small shell
script that sets up `LD_PRELOAD` so as to load the ksim shared object into the
application and then launches the application. It follows the convention of similar
launchers:

```sh
$ ksim [ksim args] [--] application [application args]
```

The ksim args may change on a whim, but typically you can pass `--trace=FLAGS`, with FLAGS
being something like `avx,gs,gem` which
makes ksim print a lot of info about the given topics.
