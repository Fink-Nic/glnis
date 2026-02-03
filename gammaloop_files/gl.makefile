NAME = test
N_CORES = 32
GL_PATH = ./target/dev-optim/gammaloop
AMPLITUDE = ./gl_states/$(NAME)/processes/amplitudes/$(NAME)
OUTPUT_STATE = ./gl_states/$(NAME)_output
INTEGRATION_STATE = ./gl_states/$(NAME)_integration
INTEGRATION_RESULTS = $(INTEGRATION_STATE)/integration_results.txt
RUNCARD = ./runcards/generate_$(NAME).toml

generate:
	echo "$(NAME)"
	rm -rf $(OUTPUT_STATE); $(GL_PATH) -o -s $(OUTPUT_STATE) -t generate $(RUNCARD)

integrate:
	echo "$(NAME)"
	rm -rf $(INTEGRATION_STATE); GL_DISPLAY_FILTER=info GL_LOGFILE_FILTER=warning $(GL_PATH) -n -s $(OUTPUT_STATE) integrate --workspace-path $(INTEGRATION_STATE) --result-path $(INTEGRATION_RESULTS) -c $(N_CORES)
