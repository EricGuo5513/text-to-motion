# Installation Notes on Mac M2

Note
- Mac M2 (ARM64) doesn't have conda support for python-3.7 (earliest supported is python-3.8)

## Correction

- install conda env
```
conda env  create -f environment.yaml
```

## Unzip into Checkpoints

```
╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  21:56:54
╰─❯ ls -lh
total 1.4G
-rw-rw-r-- 1 seki seki 672M Sep 28 22:00 kit.zip
-rw-rw-r-- 1 seki seki 674M Sep 28 22:00 t2m.zip

╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  22:02:58
╰─❯ unzip t2m.zip 
Archive:  t2m.zip
   creating: t2m/
...

╭─   ~/src/aitok/text-to-motion/checkpoints   main !1 ·································································  text2motion_pub seki@Legion-Ubuntu  22:03:50
╰─❯ unzip kit.zip 
Archive:  kit.zip
   creating: kit/
```
