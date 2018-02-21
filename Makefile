project = all_prefix_sum

bin_dir = bin
src_dir = src
srcs = $(wildcard $(src_dir)/*.cu)
executable_path = $(bin_dir)/$(project)

nvcc = nvcc
nvcc_flags = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc

remote_host ?= cca-lg04-072
remote_path ?= ~/src/all_prefix_sum

$(bin_dir):
	@mkdir -p $(bin_dir)

$(executable_path): $(srcs) | $(bin_dir)
	$(nvcc) $(nvcc_flags) -o $@ $^

.PHONY: build run remote-build remote-run remote-deploy

build: $(executable_path)

run: build
	$(executable_path) $(args)

remote-build: remote-deploy
	ssh ${remote_host} " \
		module load cuda && \
		cd $(remote_path) && \
		$(MAKE) build"

remote-run: remote-deploy
	ssh ${remote_host} " \
		module load cuda && \
		cd $(remote_path) && \
		$(MAKE) args=$(args) run"

remote-deploy:
	rsync -ra --exclude "*.git" . $(remote_host):$(remote_path)

