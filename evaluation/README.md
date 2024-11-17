Evaluation command:

```bash
cd evaluation/vos-benchmark
# J&F
python benchmark.py -g [path to gt] -m [path to predicts] --do_not_skip_first_and_last_frame
# tIoU
python tiou.py [path to predicts] spans.txt
```