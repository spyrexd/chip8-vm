use pretty_hex::pretty_hex;
use std::fmt;
use std::{ops, usize};

const MEM_SIZE: usize = 4096;

//128 x 64
//const DISPLAY_SIZE: (usize, usize) = (128, 64);
const DISPLAY_SIZE: (usize, usize) = (64, 32);

const FONT_DATA: [u8; 80] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

#[derive(Debug)]
pub enum CpuError {
    InvalidMemoryAccess(u16),
    InvalidInstruction(u16),
    EmptyStack,
    MemoryWriteError,
}

#[derive(Debug)]
pub enum Mode {
    Debug,
    Normal,
}
#[derive(Debug)]
pub struct CPU {
    pc: u16,
    idx: u16,
    v: [u8; 16],
    delay: u8,
    sound: u8,
    stack: Vec<u16>,
    mode: Mode,
    instruction: Instruction,
}

impl CPU {
    pub fn new() -> Self {
        CPU {
            pc: 0x0200,
            idx: 0x000,
            v: [0x0; 16],
            delay: 0x0,
            sound: 0x0,
            stack: vec![],
            mode: Mode::Debug,
            instruction: Instruction(0x000),
        }
    }

    pub fn reset(&mut self) {
        self.pc = 0x0200;
        self.idx = 0x000;
        self.v.copy_from_slice(&[0x00;16]);
        self.delay = 0x0;
        self.sound = 0x0;
        self.stack.clear();
        self.instruction = Instruction(0x0000);
    }

    pub fn tick(&mut self) {
        if self.delay > 0 {
            self.delay -= 1;
        }
        if self.sound > 0 {
            self.sound -= 1;
        }
    }

    pub fn run(&mut self, memory: &mut Memory) -> Result<(), CpuError> {
        loop {
            self.fetch(memory)?;
            self.execute(memory)?;
        }
    }

    pub fn fetch(&mut self, memory: &Memory) -> Result<Instruction, CpuError> {
        if self.pc >= (MEM_SIZE - 2) as u16 {
            return Err(CpuError::InvalidMemoryAccess(self.pc));
        }
        self.instruction = Instruction::read(&memory.data[(self.pc as usize)..]);
        self.pc += 2;
        Ok(self.instruction)
    }

    pub fn execute(&mut self, memory: &mut Memory) -> Result<(), CpuError> {
        match self.mode {
            Mode::Debug => println!("{:X?}", self),
            _ => {}
        }
        let instruction = self.instruction;
        let opcode = instruction.opcode();
        match opcode {
            0x0 => match instruction.byte() {
                0xE0 => { 
                    for pixel in &mut memory.vram[..] {
                        *pixel = 0x0;
                    }
                }
                0xEE => self.pc = self.stack.pop().ok_or(CpuError::EmptyStack)?,
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            0x1 => self.pc = instruction.addr(),
            0x2 => {
                self.stack.push(self.pc);
                self.pc = instruction.addr();
            }
            0x3 => {
                let value = instruction.byte();
                if self.v[instruction.x() as usize] == value {
                    self.pc += 2;
                }
            }
            0x4 => {
                
                let value = instruction.byte();
                if self.v[instruction.x() as usize] != value {
                    self.pc += 2;
                }
            }
            0x5 => match instruction.nibble() {
                0x0 => {
                    
                    if self.v[instruction.x() as usize] == self.v[instruction.y() as usize] {
                        self.pc += 2;
                    }
                }
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            0x6 => {
                
                let value = instruction.byte();
                self.v[instruction.x() as usize] = value;
            }
            0x7 => {
                
                let value = instruction.byte();
                let current_reg_value = self.v[instruction.x() as usize];
                self.v[instruction.x() as usize] = current_reg_value.wrapping_add(value);
            }
            0x8 => {
                
                match instruction.nibble() {
                    0x0 => self.v[instruction.x() as usize] = self.v[instruction.y() as usize],
                    0x1 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        self.v[instruction.x() as usize] = vx | vy;
                    }
                    0x2 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        self.v[instruction.x() as usize] =  vx & vy;
                    }
                    0x3 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        self.v[instruction.x() as usize]  = vx ^ vy;
                    }
                    0x4 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        let (add, overflow) = vx.overflowing_add(vy);
                        self.v[instruction.x() as usize] = add;
                        self.v[0xF] = overflow as u8;
                    }
                    0x5 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        self.v[0xF] =  1;
                        if vx > vy {
                            self.v[instruction.x() as usize] =  vx - vy;
                        } else {
                            self.v[instruction.x() as usize] =  vx.wrapping_sub(vy);
                            self.v[0xF] = 0;
                        }
                    }
                    0x6 => {
                        let vx = self.v[instruction.x() as usize];
                        self.v[0xF] = vx & 0x01;
                        let shr = vx >> 1;
                        self.v[instruction.x() as usize] = shr;
                    }
                    0x7 => {
                        let vx = self.v[instruction.x() as usize];
                        let vy = self.v[instruction.y() as usize];
                        self.v[0xF] =  0x01;
                        if vy > vx {
                            self.v[instruction.x() as usize] = vy - vx;
                        } else {
                            self.v[instruction.x() as usize] = vy.wrapping_sub(vx);
                            self.v[0xF] = 0x0;
                        }
                    }
                    0xE => {
                        let vx = self.v[instruction.x() as usize];
                        self.v[0xF] = (vx & 0x80) >> 7;
                        let shl = vx << 1;
                        self.v[instruction.x() as usize] = shl;
                    }
                    _ => return Err(CpuError::InvalidInstruction(instruction.0)),
                }
            }
            0x9 => match instruction.nibble() {
                0x0 => {
                    
                    if self.v[instruction.x() as usize] != self.v[instruction.y() as usize] {
                        self.pc += 2;
                    }
                }
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            0xA => {
                self.idx = instruction.addr();
            }
            0xB => {
                self.pc = instruction.addr() + self.v[0x0] as u16;
            }
            0xC => {
                use rand::prelude::*;

                let mut rng = rand::thread_rng();
                let rand_byte: u8 = rng.gen();
                let value = instruction.byte();
                self.v[instruction.x() as usize] = value & rand_byte;
            }
            0xD => {
                self.v[0xF] = 0x0;
                let num_rows = instruction.nibble();
                let start_x  = self.v[instruction.x() as usize] % DISPLAY_SIZE.0 as u8;
                let start_y = self.v[instruction.y() as usize] % DISPLAY_SIZE.1 as u8;
                'row: for row in 0..num_rows {
                    if start_y + row >= DISPLAY_SIZE.1 as u8 {
                        break 'row;
                    }
                    let sprite_data = memory[self.idx + row as u16];
                    for col in 0..8 {
                        if col + start_x >= DISPLAY_SIZE.0 as u8 {
                            continue 'row;
                        }
                        let pos: u16 = (start_x + col) as u16 + (start_y + row) as u16 * 64u16;
                        if sprite_data & (0x80 >> col) != 0 {
                            if memory.vram[pos as usize] == 0x1 {
                                self.v[0xF] = 0x1;
                                memory.vram[pos as usize] = 0x0;
                            } else {
                                memory.vram[pos as usize] = 0x1;
                            }
                        }
                    }
                }
                
            }
            0xE => {
                let vx = self.v[instruction.x() as usize] as usize; 
                match instruction.byte() {
                    0x9E => {
                        if memory.kb[vx] == 0x01 {
                            self.pc += 2;
                        }
                    }
                    0xA1 => {
                        if memory.kb[vx] == 0x00 {
                            self.pc += 2;
                        }
                    }
                    _ => return Err(CpuError::InvalidInstruction(instruction.0)),
                }
            }
            0xF => {
                
                match instruction.byte() {
                    0x07 => self.v[instruction.x() as usize] = self.delay,
                    0x15 => self.delay = self.v[instruction.x() as usize],
                    0x18 => self.sound = self.v[instruction.x() as usize],
                    0x1E => {
                        let add = self.idx.overflowing_add(self.v[instruction.x() as usize] as u16);
                        self.idx = add.0;
                        self.v[0xF] = add.1 as u8;
                    }
                    0x0A => {
                      if let Some(p) = memory.kb.iter().position(|&x| x == 0x01) {
                          self.v[instruction.x() as usize] = p as u8;
                      } else {
                          self.pc -= 2;
                      }
                       
                    }
                    0x29 => {
                        let fc = self.v[instruction.x() as usize];
                        self.idx = 0x50u16 + ((fc & 0x0F) as u16 * 5u16);
                    }
                    0x33 => {
                        let num = self.v[instruction.x() as usize];
                        let hundreds = num / 100u8;
                        let tens = (num % 100u8) / 10u8;
                        let ones = num - (hundreds * 100) - (tens * 10);
                        memory[self.idx] = hundreds;
                        memory[self.idx + 1u16] = tens;
                        memory[self.idx + 2u16] = ones;
                    }
                    0x55 => {
                        for i in 0..(instruction.x() as u8 + 1) {
                            memory[self.idx + i as u16] = self.v[i as usize];
                        }
                    }
                    0x65 => {
                        for i in 0..(instruction.x() as u8 + 1) {
                            self.v[i as usize] = memory[self.idx + i as u16];
                        }
                    }
                    _ => return Err(CpuError::InvalidInstruction(instruction.0)),
                }
            }
            _ => return Err(CpuError::InvalidInstruction(instruction.0)),
        }
        Ok(())
    }

    pub fn load_font(&self, memory: &mut Memory) -> Result<(), CpuError> {
        memory.load(&FONT_DATA, 0x50)
    }
}

#[derive(PartialEq, Clone, Copy)]
pub struct Instruction(u16);

impl Instruction {
    pub fn read(memory: &[u8]) -> Self {
        Instruction(((memory[0] as u16) << 8) | memory[1] as u16)
    }

    fn opcode(&self) -> u8 {
        (self.0 >> 12 & 0x0F) as u8
    }

    fn x(&self) -> u8 {
        (self.0 >> 8 & 0x0F) as u8
    }

    fn y(&self) -> u8 {
        (self.0 >> 4 & 0x0F) as u8
    }

    fn nibble(&self) -> u8 {
        (self.0 & 0x000F) as u8
    }

    fn byte(&self) -> u8 {
        (self.0 & 0x00FF) as u8
    }

    fn addr(&self) -> u16 {
        self.0 & 0x0FFF
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instruction(0x{:04X})", self.0)
    }
}
pub struct Memory {
    data: Vec<u8>,
    vram: Vec<u8>,
    kb: Vec<u8>
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            data: vec![0u8; MEM_SIZE],
            vram: vec![0u8; DISPLAY_SIZE.0*DISPLAY_SIZE.1],
            kb: vec![0u8; 16],
        }
    }

    pub fn reset_progam_memory(&mut self) {
        for addr in &mut self.data {
            *addr = 0u8;
        }
    }

    pub fn reset_vram(&mut self) {
        for p in &mut self.vram {
            *p = 0x0;
        }
    }

    pub fn reset_kb(&mut self) {
        for k in &mut self.kb {
            *k = 0x0;
        }
    }

    pub fn reset_all(&mut self) {
        self.reset_progam_memory();
        self.reset_vram();
        self.reset_kb();
    }


    pub fn load(&mut self, data: &[u8], start: usize) -> Result<(), CpuError> {
        if start + data.len() > MEM_SIZE {
            return Err(CpuError::MemoryWriteError);
        }
        // Find better way
        for (i, v) in data.iter().enumerate() {
            self[(start + i) as u16] = *v;
        }
        Ok(())
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, v) in self.vram.iter().enumerate() {
            if i %  DISPLAY_SIZE.0 == 0 && i > 0 {
                write!(f, "\n")?;
            }
            if *v == 0x0 {
                write!(f, " ")?;
            } else {
                write!(f, "\u{2588}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{}", pretty_hex(&self.data)))
    }
}

impl ops::Index<u16> for Memory {
    type Output = u8;

    fn index(&self, index: u16) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl ops::IndexMut<u16> for Memory {
    fn index_mut(&mut self, index: u16) -> &mut Self::Output {
        &mut self.data[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_tests() {
        let mut mem = Memory::new();
        let address: u16 = 0x42;
        assert_eq!(mem[address], 0x00);
        mem[address] = 0x42;
        assert_eq!(mem[address], 0x42);
        mem.reset_all();
        assert_eq!(mem[address], 0x00);
    }

    #[test]
    fn instruction_tests() {
        let ins = Instruction(0x1123);
        assert_eq!(ins.opcode(), 0x01);
        assert_eq!(ins.x(), 0x1);
        assert_eq!(ins.y(), 0x2);
        assert_eq!(ins.byte(), 0x23);
        assert_eq!(ins.nibble(), 0x03);
        assert_eq!(ins.addr(), 0x0123);
    }
    #[test]
    fn create_cpu() {
        let cpu = CPU::new();
        assert_eq!(cpu.pc, 0x200);
    }

    #[test]
    fn load_font() {
        let cpu = CPU::new();
        let mut mem = Memory::new();
        let res = cpu.load_font(&mut mem);
        assert!(res.is_ok());
    }

    #[test]
    fn fetch_instruction() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        mem[cpu.pc] = 0xFF;
        mem[cpu.pc + 1] = 0xFF;
        assert_eq!(cpu.fetch(&mem).unwrap(), Instruction(0xFFFF))
    }

    #[test]
    fn fetch_instruction_invalid_address() {
        let mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.pc = 0xFFFF;
        let ins = cpu.fetch(&mem);
        assert!(ins.is_err());
    }

    //0x0000
    #[test]
    fn instruction_cls() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x00E0);
        let _ = cpu.execute(&mut mem);
    }

    #[test]
    fn instruction_ret() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x00EE);
        cpu.stack.push(0x0123);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x1000
    #[test]
    fn instruction_jump() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x1123);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x2000
    #[test]
    fn instruction_call() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x2123);
        let _ = cpu.execute(&mut mem);
        assert_eq!(*(cpu.stack.last().unwrap()), 0x0200);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x3000
    #[test]
    fn instruction_skip_equal_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x3042);
        let start_pc = cpu.pc;
        cpu.v[0x0] = 0x42;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    //0x4000
    #[test]
    fn instruction_skip_not_equal_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x4042);
        let start_pc = cpu.pc;
        cpu.v[0x0] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    //0x5000
    #[test]
    fn instruction_skip_reg_equal() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x5010);
        let start_pc = cpu.pc;
        cpu.v[0x0] = 0x42;
        cpu.v[0x1] = 0x42;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    #[test]
    fn instruction_skip_reg_equal_fail() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x501F);
        let res = cpu.execute(&mut mem);
        assert!(res.is_err());
    }

    //0x6000
    #[test]
    fn instruction_ld_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x60FF);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFF);
    }

    //0x7000
    #[test]
    fn instruction_add() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x70FF);
        cpu.v[0x0] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x00);
    }

    //0X8000
    #[test]
    fn instruction_ld_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8010);
        cpu.v[0x0] = 0x00;
        cpu.v[0x1] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], cpu.v[0x1]);
    }

    #[test]
    fn instruction_or_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8011);
        cpu.v[0x0] = 0x0F;
        cpu.v[0x1] = 0xF0;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFF);
    }

    #[test]
    fn instruction_xor_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8013);
        cpu.v[0x0] = 0xEF;
        cpu.v[0x1] = 0xFE;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x11);
    }

    #[test]
    fn instruction_add_with_carry() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8014);
        cpu.v[0x0] = 0xFF;
        cpu.v[0x1] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x00);
        assert_eq!(cpu.v[0xF], 0x01);


        cpu.v[0x0] = 0x41;
        cpu.v[0x1] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x42);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    #[test]
    fn instruction_sub_with_barrow() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8015);
        cpu.v[0x0] = 0xFF;
        cpu.v[0x1] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFE);
        assert_eq!(cpu.v[0xF], 0x01);

        cpu.v[0x0] = 0x00;
        cpu.v[0x1] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFF);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    #[test]
    fn instruction_shr() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8006);
        cpu.v[0x0] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x7F);
        assert_eq!(cpu.v[0xF], 0x01);

        cpu.v[0x0] = 0x02;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x01);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    #[test]
    fn instruction_subn() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x8017);
        cpu.v[0x0] = 0x01;
        cpu.v[0x1] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFE);
        assert_eq!(cpu.v[0xF], 0x01);

        cpu.v[0x0] = 0x01;
        cpu.v[0x1] = 0x00;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFF);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    #[test]
    fn instruction_shl() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x800E);
        cpu.v[0x0] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0xFE);
        assert_eq!(cpu.v[0xF], 0x01);

        cpu.v[0x0] = 0x01;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x02);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    //0x9000
    #[test]
    fn instruction_skip_reg_not_equal() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x9010);
        let start_pc = cpu.pc;
        cpu.v[0x0] = 0x42;
        cpu.v[0x1] = 0x84;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, start_pc + 2);
    }
    #[test]
    fn instruction_skip_reg_not_equal_fail() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0x9011);
        cpu.v[0x0] = 0x42;
        cpu.v[0x1] = 0x84;
        let res = cpu.execute(&mut mem);
        assert!(res.is_err());
    }

    //0x0A000
    #[test]
    fn instruction_ld_idx() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xAFFF);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.idx, 0x0FFF);
    }

    //0xB000
    #[test]
    fn instruction_jmp_v0() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xB200);
        cpu.v[0x0] = 0x42;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.pc, 0x0242);
    }

    //0xC000
    #[test]
    fn instruction_reg_rnd() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xC0FF);
        let res = cpu.execute(&mut mem);
        assert!(res.is_ok());
    }

    //0xD0000
    #[test]
    fn logo() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let data = [
            0x00, 0xE0, 0xA2, 0x2A, 0x60, 0x0C, 0x61, 0x08, 0xD0, 0x1F, 0x70, 0x09, 0xA2, 0x39,
            0xD0, 0x1F, 0xA2, 0x48, 0x70, 0x08, 0xD0, 0x1F, 0x70, 0x04, 0xA2, 0x57, 0xD0, 0x1F,
            0x70, 0x08, 0xA2, 0x66, 0xD0, 0x1F, 0x70, 0x08, 0xA2, 0x75, 0xD0, 0x1F, 0x12, 0x28,
            0xFF, 0x00, 0xFF, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0xFF, 0x00,
            0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x38, 0x00, 0x3F, 0x00, 0x3F, 0x00, 0x38, 0x00, 0xFF,
            0x00, 0xFF, 0x80, 0x00, 0xE0, 0x00, 0xE0, 0x00, 0x80, 0x00, 0x80, 0x00, 0xE0, 0x00,
            0xE0, 0x00, 0x80, 0xF8, 0x00, 0xFC, 0x00, 0x3E, 0x00, 0x3F, 0x00, 0x3B, 0x00, 0x39,
            0x00, 0xF8, 0x00, 0xF8, 0x03, 0x00, 0x07, 0x00, 0x0F, 0x00, 0xBF, 0x00, 0xFB, 0x00,
            0xF3, 0x00, 0xE3, 0x00, 0x43, 0xE0, 0x00, 0xE0, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
            0x00, 0x80, 0x00, 0xE0, 0x00, 0xE0,
        ];
        let _ = mem.load(&data, cpu.pc as usize);
        for _i in 0..25 {
            let _ = cpu.fetch(&mem).unwrap();
            let _ = cpu.execute(&mut mem);
        }
        println!("{}", mem);
    }

    //OxE0000
    #[test]
    fn skip_next_instruction_key_pressed() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xE09E);
        cpu.v[0x0] = 0x00;
        mem.kb[0x0] = 0x01;
        let pc = cpu.pc;
        let _ = cpu.execute(&mut mem);
        assert_eq!(pc + 2, cpu.pc);

    }

    #[test]
    fn skip_next_instruction_key_not_pressed() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xE0A1);
        cpu.v[0x0] = 0x00;
        mem.kb[0x0] = 0x00;
        let pc = cpu.pc;
        let _ = cpu.execute(&mut mem);
        assert_eq!(pc + 2, cpu.pc);
    }

    //0xF000
    #[test]
    fn ld_dt_ref() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.delay = 0x42;
        cpu.instruction = Instruction(0xF007);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x42);
    }

    #[test]
    fn wait_for_key_press() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut pc = cpu.pc;
        mem.kb[0xF] = 0x01;
        mem[pc] = 0xFF;
        mem[pc +1 ] = 0x0A;
        let _ = cpu.fetch(&mem);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0xF], 0xF);
        assert_eq!(pc + 2, cpu.pc);

        pc = cpu.pc;
        mem[pc] = 0xFF;
        mem[pc +1 ] = 0x0A;
        mem.kb[0xF] = 0x00;
        cpu.v[0xF] = 0x00;
        let _ = cpu.fetch(&mem);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0xF], 0x00);
        assert_eq!(pc, cpu.pc);


    }

    #[test]
    fn set_delay_timer() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.v[0x0] = 0x42;
        cpu.instruction = Instruction(0xF015);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.delay, 0x42);
    }

    #[test]
    fn set_sound_timer() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.v[0x0] = 0x42;
        cpu.instruction = Instruction(0xF018);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.sound, 0x42);
    }

    #[test]
    fn idx_add_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.v[0x0] = 0x42;
        cpu.idx = 0x0001;
        cpu.instruction = Instruction(0xF01E);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.idx, 0x0043);
        assert_eq!(cpu.v[0xF], 0x00);
    }

    #[test]
    fn set_idx_to_digit() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.v[0x0] = 0x42;
        cpu.instruction = Instruction(0xF029);
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.idx, 0x5A);
    }

    #[test]
    fn store_bcd_at_idx() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xF033);
        cpu.v[0x0] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(mem[cpu.idx], 2 as u8);
        assert_eq!(mem[cpu.idx + 1], 5 as u8);
        assert_eq!(mem[cpu.idx + 2], 5 as u8);
    }

    #[test]
    fn store_all_reg_from_idx() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xF155);
        cpu.v[0x0] = 0x01;
        cpu.v[0x1] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(mem[cpu.idx], 0x01);
        assert_eq!(mem[cpu.idx + 0x0001], 0xFF);
    }

    #[test]
    fn load_all_reg_from_idx() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        cpu.instruction = Instruction(0xF165);
        mem[cpu.idx] = 0x01;
        mem[cpu.idx + 0x1] = 0xFF;
        let _ = cpu.execute(&mut mem);
        assert_eq!(cpu.v[0x0], 0x01);
        assert_eq!(cpu.v[0x1], 0xFF);
    }
}
