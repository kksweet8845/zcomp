GCC:=gcc

C_FLAGS:= -lm -mavx -mavx2 -mavx512f -mavx512vl

INC_DIR:=../common
INC:=$(wildcard ../common/*.c)
INC_O:=$(patsubst %.c, %.o, $(INC))


%.o: %.c
	$(GCC) $(C_FLAGS) -c -o $@ $^ -I../common


zcomp: zcomp-test.o ../common/zcomp.o
	$(GCC) $(C_FLAGS) -o $@ $^