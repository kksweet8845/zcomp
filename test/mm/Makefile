TARGET=mm
include ../shared/Makefile.shared

$(TARGET): $(TARGET).o
	$(CC) $(TARGET).o -lm $(SNIPER_LDFLAGS) -o $(TARGET)


run_$(TARGET):
	../../run-sniper -v -n 1 -c gainestown --memory-profile --roi -- ./$(TARGET) -p 1

CLEAN_EXTRA=viz


