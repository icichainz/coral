# coral — high-performance LLM inference engine for Apple Silicon GPUs.
#
# Language policy:
#   * Engine: standard C++20.  One Objective-C++ file (src/gpu/metal_backend.mm)
#     owns every Metal call behind the plain C++ interface in include/coral/gpu.h.
#   * Kernels: Metal Shading Language, embedded as string literals at build time
#     and compiled by the Metal runtime on first launch.
#   * Python appears only under scripts/ and tests/reference/ as dev tooling.
#
# Targets:
#   make            build build/coral and build/coral_tests
#   make test       build and run the unit tests
#   make run ARGS=  run the binary with arguments
#   make clean

CXX        := clang++
BUILD      := build
GEN        := $(BUILD)/gen
MACOS_MIN  := 26.0

CXXFLAGS   := -std=c++20 -O2 -g -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
              -fvisibility=hidden -mmacosx-version-min=$(MACOS_MIN) \
              -Iinclude -I$(GEN)
OBJCXXFLAGS:= $(CXXFLAGS) -fobjc-arc -x objective-c++
LDFLAGS    := -framework Metal -framework Foundation -mmacosx-version-min=$(MACOS_MIN)

ifeq ($(DEBUG),1)
  CXXFLAGS := $(filter-out -O2,$(CXXFLAGS)) -O0 -DCORAL_DEBUG=1 -fsanitize=address
  LDFLAGS  += -fsanitize=address
endif

# ---- sources ---------------------------------------------------------------
CPP_SRCS := $(shell find src -name '*.cpp' -not -name 'main.cpp')
MM_SRCS  := $(shell find src -name '*.mm')
KERNELS  := $(sort $(shell find src/kernels -name '*.metal' -o -name '*.h'))

CPP_OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(CPP_SRCS))
MM_OBJS  := $(patsubst src/%.mm,$(BUILD)/obj/%.o,$(MM_SRCS))
LIB_OBJS := $(CPP_OBJS) $(MM_OBJS)

TEST_SRCS := $(shell find tests -name '*.cpp')
TEST_OBJS := $(patsubst tests/%.cpp,$(BUILD)/obj/tests/%.o,$(TEST_SRCS))

SHADERS_INC := $(GEN)/shaders.inc
EMBED_TOOL  := $(BUILD)/embed_shaders

.PHONY: all test run clean gen
all: $(BUILD)/coral $(BUILD)/coral_tests

gen: $(SHADERS_INC)

# ---- shader embedding ------------------------------------------------------
$(EMBED_TOOL): tools/embed_shaders.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++20 -O2 -o $@ $<

$(SHADERS_INC): $(EMBED_TOOL) $(KERNELS)
	@mkdir -p $(dir $@)
	$(EMBED_TOOL) $@ $(KERNELS)

# ---- compilation -----------------------------------------------------------
$(BUILD)/obj/%.o: src/%.cpp $(SHADERS_INC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/obj/%.o: src/%.mm $(SHADERS_INC)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/obj/tests/%.o: tests/%.cpp $(SHADERS_INC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Itests -MMD -MP -c $< -o $@

# ---- link ------------------------------------------------------------------
$(BUILD)/coral: $(LIB_OBJS) $(BUILD)/obj/main.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD)/coral_tests: $(LIB_OBJS) $(TEST_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

test: $(BUILD)/coral_tests
	./$(BUILD)/coral_tests

run: $(BUILD)/coral
	./$(BUILD)/coral $(ARGS)

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJS:.o=.d) $(BUILD)/obj/main.d $(TEST_OBJS:.o=.d)
