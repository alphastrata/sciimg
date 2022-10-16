baseline (Scott's implementation of hot_pxl)
	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     693.1 ms ±  11.4 ms    [User: 670.4 ms, System: 22.9 ms]
	  Range (min … max):   675.8 ms … 704.6 ms    10 runs


	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     686.9 ms ±   9.6 ms    [User: 664.9 ms, System: 22.1 ms]
	  Range (min … max):   675.3 ms … 706.8 ms    10 runs

nested for loops into .iter().map(.....)
	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     722.8 ms ±   8.0 ms    [User: 700.8 ms, System: 22.1 ms]
	  Range (min … max):   706.2 ms … 737.2 ms    10 runs


	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     708.4 ms ±  13.1 ms    [User: 686.8 ms, System: 21.7 ms]
	  Range (min … max):   688.7 ms … 728.0 ms    10 runs


outer loop into par_iter() collections mutexed, replacement_pixels collected rather than ITM

	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     570.3 ms ±   6.5 ms    [User: 2264.0 ms, System: 12476.6 ms]
	  Range (min … max):   562.5 ms … 584.0 ms    10 runs


	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     567.8 ms ±   4.1 ms    [User: 2329.4 ms, System: 12476.5 ms]
	  Range (min … max):   560.9 ms … 573.7 ms    10 runs


	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     565.8 ms ±   7.8 ms    [User: 2288.0 ms, System: 12178.3 ms]
	  Range (min … max):   547.0 ms … 574.5 ms    10 runs

spsc (expected to be terrible)
Benchmark 1: cargo test --release test_hot_pixel_correction
  Time (mean ± σ):      2.155 s ±  0.187 s    [User: 3.023 s, System: 0.794 s]
  Range (min … max):    1.919 s …  2.462 s    10 runs


Benchmark 1: cargo test --release test_hot_pixel_correction
  Time (mean ± σ):      2.239 s ±  0.079 s    [User: 2.946 s, System: 0.759 s]
  Range (min … max):    2.119 s …  2.379 s    10 runs


mpsc (should be Ok) -- unable to get this one working ><


isolate_window() nested for loops vs iterators
-  rare case where the size of the buffer is known upfront.
- in _this_ instance the nested for-loops are fine
- in realistic use-cases the lazy approach may still be preferable.
for-loops:
	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     585.3 ms ±   4.9 ms    [User: 2404.1 ms, System: 12971.5 ms]
	  Range (min … max):   575.6 ms … 599.9 ms    50 runs

Iterators:
	Benchmark 1: cargo test --release test_hot_pixel_correction
	  Time (mean ± σ):     610.9 ms ±   4.0 ms    [User: 5328.9 ms, System: 10759.2 ms]
	  Range (min … max):   602.7 ms … 618.7 ms    50 runs
