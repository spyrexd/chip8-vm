use bitvec::prelude::*;
use derive_try_from_primitive::TryFromPrimitive;
use pretty_hex::pretty_hex;
use std::{convert::TryFrom, fmt};
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

pub struct Display {
    x: usize,
    y: usize,
    data: Vec<u8>,
}

impl fmt::Display for Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, v) in self.data.iter().enumerate() {
            if i % self.x == 0  && i > 0{
                write!(f, "\n" )?;
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

impl fmt::Debug for Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{}", pretty_hex(&self.data)))
    }
}

impl Display {
    fn new() -> Self {
        Display {
            x: DISPLAY_SIZE.0,
            y: DISPLAY_SIZE.1,
            data: vec![0x0; DISPLAY_SIZE.0 * DISPLAY_SIZE.1],
        }
    }

    fn cls(&mut self) {
        for addr in self.data.as_mut_slice() {
            *addr = 0x0;
        }
    }

    fn draw(&mut self, x: &u8, y: &u8, values: &[u8]) -> bool {
        let mut collision = false;
        'row: for (i, v) in values.iter().enumerate() {
            let row = (*y as usize + i) * self.x;
            if row >= self.x * self.y {
                break 'row;
            }
            let bits = BitSlice::<Msb0, _>::from_element(v);
            for (j, b) in bits.iter().enumerate() {
                let col = *x as usize + j;
                if col >= self.x {
                    continue 'row;
                }
                let location = row + col;
                let current = self.data[location];
                let xor = current ^ *b as u8;
                if current == 0x1 && xor == 0x0 {
                    collision = true;
                }
                self.data[location] = xor;
            }
        }
        collision
    }
}

#[derive(Debug)]
pub enum CpuError {
    InvalidMemoryAccess(u16),
    InvalidRegister(<Register as TryFrom<u8>>::Error),
    InvalidOpCode(<OpCode as TryFrom<u16>>::Error),
    InvalidInstruction(u16),
    EmptyStack,
    MemoryWriteError,
}

impl From<<Register as TryFrom<u8>>::Error> for CpuError {
    fn from(err: <Register as TryFrom<u8>>::Error) -> CpuError {
        CpuError::InvalidRegister(err)
    }
}

impl From<<OpCode as TryFrom<u16>>::Error> for CpuError {
    fn from(err: <OpCode as TryFrom<u16>>::Error) -> CpuError {
        CpuError::InvalidOpCode(err)
    }
}

#[derive(Debug, TryFromPrimitive, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum Register {
    V0 = 0x0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    VA,
    VB,
    VC,
    VD,
    VE,
    VF,
}

#[derive(Debug, TryFromPrimitive, PartialEq)]
#[repr(u16)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    OC_0 = 0x0000,
    OC_1 = 0x1000,
    OC_2 = 0x2000,
    OC_3 = 0x3000,
    OC_4 = 0x4000,
    OC_5 = 0x5000,
    OC_6 = 0x6000,
    OC_7 = 0x7000,
    OC_8 = 0x8000,
    OC_9 = 0x9000,
    OC_A = 0xA000,
    OC_B = 0xB000,
    OC_C = 0xC000,
    OC_D = 0xD000,
    OC_E = 0xE000,
    OC_F = 0xF000,
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
    registers: [u8; 16],
    delay: u8,
    sound: u8,
    stack: Vec<u16>,
    mode: Mode,
    instruction: Instruction,
}

impl CPU {
    fn new() -> Self {
        CPU {
            pc: 0x0200,
            idx: 0x000,
            registers: [0x0; 16],
            delay: 0x0,
            sound: 0x0,
            stack: vec![],
            mode: Mode::Debug,
            instruction: Instruction(0x000),
        }
    }

    fn set_reg(&mut self, reg: Register, value: u8) {
        self.registers[reg as usize] = value;
    }

    fn get_reg(&self, reg: Register) -> u8 {
        self.registers[reg as usize]
    }

    fn fetch(&mut self, memory: &Memory) -> Result<Instruction, CpuError> {
        if self.pc >= MEM_SIZE as u16 {
            return Err(CpuError::InvalidMemoryAccess(self.pc));
        }
        let ins = Instruction(((memory[self.pc] as u16) << 8) | memory[self.pc + 1] as u16);
        self.pc += 2;
        self.instruction = ins;
        Ok(ins)
    }

    fn execute(
        &mut self,
        memory: &mut Memory,
        display: &mut Display,
        instruction: Instruction,
    ) -> Result<(), CpuError> {
        match self.mode {
            Mode::Debug => println!("{:X?}", self),
            _ => {},
        }
        let opcode = instruction.opcode()?;
        match opcode {
            OpCode::OC_0 => match instruction.low_byte() {
                0xE0 => display.cls(),
                0xEE => self.pc = self.stack.pop().ok_or(CpuError::EmptyStack)?,
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            OpCode::OC_1 => self.pc = instruction.address(),
            OpCode::OC_2 => {
                self.stack.push(self.pc);
                self.pc = instruction.address();
            }
            OpCode::OC_3 => {
                let registers = instruction.registers()?;
                let value = instruction.low_byte();
                if self.get_reg(registers.0) == value {
                    self.pc += 2;
                }
            }
            OpCode::OC_4 => {
                let registers = instruction.registers()?;
                let value = instruction.low_byte();
                if self.get_reg(registers.0) != value {
                    self.pc += 2;
                }
            }
            OpCode::OC_5 => match instruction.last_nibble() {
                0x0 => {
                    let registers = instruction.registers()?;
                    if self.get_reg(registers.0) == self.get_reg(registers.1) {
                        self.pc += 2;
                    }
                }
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            OpCode::OC_6 => {
                let registers = instruction.registers()?;
                let value = instruction.low_byte();
                self.set_reg(registers.0, value);
            },
            OpCode::OC_7 => {
                let registers = instruction.registers()?;
                let value = instruction.low_byte();
                let current_reg_value = self.get_reg(registers.0);
                self.set_reg(registers.0, current_reg_value.wrapping_add(value));
            },
            OpCode::OC_8 => {
                let registers = instruction.registers()?;
                match instruction.last_nibble() {
                    0x0 => self.set_reg(registers.0, self.get_reg(registers.1)),
                    0x1 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        self.set_reg(registers.0, vx | vy);
                    }, 
                    0x2 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        self.set_reg(registers.0, vx & vy);
                    }, 
                    0x3 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        self.set_reg(registers.0, vx ^ vy);
                    }, 
                    0x4 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        let (add, overflow) = vx.overflowing_add(vy);
                        self.set_reg(registers.0, add);
                        self.set_reg(Register::VF, overflow as u8);
                    }, 
                    0x5 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        self.set_reg(Register::VF, 1);
                        if vx > vy {
                            self.set_reg(registers.0, vx - vy);
                        } else {
                            self.set_reg(registers.0, vx.wrapping_sub(vy));
                            self.set_reg(Register::VF, 0);
                        }
                    }, 
                    0x6 =>  { 
                        let vx = self.get_reg(registers.0);
                        self.set_reg(Register::VF, vx & 0x01);
                        let shr  = vx >> 1;
                        self.set_reg(registers.0, shr);
                    }, 
                    0x7 =>  { 
                        let vx = self.get_reg(registers.0);
                        let vy = self.get_reg(registers.1);
                        self.set_reg(Register::VF, 1);
                        if vy > vx {
                            self.set_reg(registers.0, vy - vx);
                        } else {
                            self.set_reg(registers.0, vy.wrapping_sub(vx));
                            self.set_reg(Register::VF, 0);
                        }
                    },
                    0xE =>  { 
                        let vx = self.get_reg(registers.0);
                        self.set_reg(Register::VF, (vx & 0x80) >> 7);
                        let shl  = vx << 1;
                        self.set_reg(registers.0, shl);
                    }, 
                    _ => return Err(CpuError::InvalidInstruction(instruction.0)),
                }
            },
            OpCode::OC_9 => match instruction.last_nibble(){
                0x0 => {
                    let registers = instruction.registers()?;
                    if self.get_reg(registers.0) != self.get_reg(registers.1) {
                        self.pc += 2;
                    }
                }
                _ => return Err(CpuError::InvalidInstruction(instruction.0)),
            },
            OpCode::OC_A => {
                self.idx = instruction.address();
            }
            OpCode::OC_B => {
                self.pc = instruction.address() + self.get_reg(Register::V0) as u16;
            }
            OpCode::OC_C => {
                unimplemented!("OpCode not handled")
            }
            OpCode::OC_D => {
                let registers = instruction.registers()?;
                let value = instruction.last_nibble() as usize;
                let x = self.get_reg(registers.0) % display.x as u8;
                let y = self.get_reg(registers.1) % display.y as u8;
                self.set_reg(Register::VF, 0x0);

                let collision = display.draw(&x, &y, &memory.data[self.idx as usize..self.idx as usize+value]);
                self.set_reg(Register::VF, collision as u8);
            }
            OpCode::OC_E => {
                unimplemented!("OpCode not handled")
            }
            OpCode::OC_F => {
                unimplemented!("OpCode not handled")
            }
        }
        Ok(())
    }

    fn load_font(&self, memory: &mut Memory, font_data: &[u8]) -> Result<(), CpuError> {
        memory.load(&font_data, 0x50)
    }
}

#[derive(PartialEq, Clone, Copy)]
struct Instruction(u16);

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instruction(0x{:04X})", self.0)
        
    }
}

impl Instruction {
    fn opcode(&self) -> Result<OpCode, CpuError> {
        Ok(OpCode::try_from(self.0 & 0xF000)?)
    }

    fn address(&self) -> u16 {
        self.0 & 0x0FFF
    }

    fn low_byte(&self) -> u8 {
        (self.0 & 0x00FF) as u8
    }

    fn registers(&self) -> Result<(Register, Register), CpuError> {
        let first = (self.0 & 0x0F00) >> 8;
        let second = (self.0 & 0x00F0) >> 4;
        Ok((
            Register::try_from(first as u8)?,
            Register::try_from(second as u8)?,
        ))
    }

    fn last_nibble(&self) -> u8 {
        (self.0 & 0x000F) as u8
    }
}

pub struct Memory {
    data: Vec<u8>,
}

impl Memory {
    fn new() -> Self {
        Memory {
            data: vec![0u8; MEM_SIZE],
        }
    }

    fn clear(&mut self) {
        for addr in self.data.as_mut_slice() {
            *addr = 0u8;
        }
    }

    fn load(&mut self, data: &[u8], start: usize) -> Result<(), CpuError> {
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
        mem.clear();
        assert_eq!(mem[address], 0x00);
    }

    #[test]
    fn instruction_tests() {
        let ins = Instruction(0x1123);
        assert_eq!(ins.opcode().unwrap(), OpCode::OC_1);
        assert_eq!(ins.registers().unwrap(), (Register::V1, Register::V2));
        assert_eq!(ins.address(), 0x0123);
    }
    #[test]
    fn create_cpu() {
        let cpu = CPU::new();
        assert_eq!(cpu.pc, 0x200);
    }

    #[test]
    fn set_and_get_reg() {
        let mut cpu = CPU::new();
        cpu.set_reg(Register::V0, 0x42);
        assert_eq!(cpu.registers[0], 0x42);
        assert_eq!(cpu.get_reg(Register::V0), 0x42);
    }

    #[test]
    fn load_font() {
        let cpu = CPU::new();
        let mut mem = Memory::new();
        let res = cpu.load_font(&mut mem, &FONT_DATA);
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
        let mut display = Display::new();
        let ins = Instruction(0x00E0);
        let _ = cpu.execute(&mut mem, &mut display, ins);
    }

    #[test]
    fn instruction_ret() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x00EE);
        cpu.stack.push(0x0123);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x1000
    #[test]
    fn instruction_jump() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x1123);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x2000
    #[test]
    fn instruction_call() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x2123);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(*(cpu.stack.last().unwrap()), 0x0200);
        assert_eq!(cpu.pc, 0x0123);
    }

    //0x3000
    #[test]
    fn instruction_skip_equal_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x3042);
        let start_pc = cpu.pc;
        cpu.set_reg(Register::V0, 0x42);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    //0x4000
    #[test]
    fn instruction_skip_not_equal_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x4042);
        let start_pc = cpu.pc;
        cpu.set_reg(Register::V0, 0xFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    //0x5000
    #[test]
    fn instruction_skip_reg_equal() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x5010);
        let start_pc = cpu.pc;
        cpu.set_reg(Register::V0, 0x42);
        cpu.set_reg(Register::V1, 0x42);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, start_pc + 2);
    }

    #[test]
    fn instruction_skip_reg_equal_fail() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x501F);
        let res = cpu.execute(&mut mem, &mut display, ins);
        assert!(res.is_err());
    }

    //0x6000
    #[test]
    fn instruction_ld_byte() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x60FF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFF);
    }

    //0x7000
    #[test]
    fn instruction_add() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x70FF);
        cpu.set_reg(Register::V0, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x00);
    }

    //0X8000
    #[test]
    fn instruction_ld_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8010);
        cpu.set_reg(Register::V0, 0x00);
        cpu.set_reg(Register::V1, 0xFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), cpu.get_reg(Register::V1));
    }

    #[test]
    fn instruction_or_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8011);
        cpu.set_reg(Register::V0, 0x0F);
        cpu.set_reg(Register::V1, 0xF0);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFF);
    }

    #[test]
    fn instruction_xor_reg() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8013);
        cpu.set_reg(Register::V0, 0xEF);
        cpu.set_reg(Register::V1, 0xFE);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x11);
    }

    #[test]
    fn instruction_add_with_carry() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8014);
        cpu.set_reg(Register::V0, 0xFF);
        cpu.set_reg(Register::V1, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x00);
        assert_eq!(cpu.get_reg(Register::VF), 0x01);

        cpu.set_reg(Register::V0, 0x41);
        cpu.set_reg(Register::V1, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x42);
        assert_eq!(cpu.get_reg(Register::VF), 0x00);
    }

    #[test]
    fn instruction_sub_with_barrow() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8015);
        cpu.set_reg(Register::V0, 0xFF);
        cpu.set_reg(Register::V1, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFE);
        assert_eq!(cpu.get_reg(Register::VF), 0x01);

        cpu.set_reg(Register::V0, 0x00);
        cpu.set_reg(Register::V1, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFF);
        assert_eq!(cpu.get_reg(Register::VF), 0x00);
    }

    #[test]
    fn instruction_shr() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8006);
        cpu.set_reg(Register::V0, 0xFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x7F);
        assert_eq!(cpu.get_reg(Register::VF), 0x01);
        
        cpu.set_reg(Register::V0, 0x02);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x01);
        assert_eq!(cpu.get_reg(Register::VF), 0x00);
    }

    #[test]
    fn instruction_subn() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x8017);
        cpu.set_reg(Register::V0, 0x01);
        cpu.set_reg(Register::V1, 0xFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFE);
        assert_eq!(cpu.get_reg(Register::VF), 0x01);

        cpu.set_reg(Register::V0, 0x01);
        cpu.set_reg(Register::V1, 0x00);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0xFF);
        assert_eq!(cpu.get_reg(Register::VF), 0x00);
  
    }

    #[test]
    fn instruction_shl() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x800E);
        cpu.set_reg(Register::V0, 0xFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        println!("{:?}", cpu);
        assert_eq!(cpu.get_reg(Register::V0), 0xFE);
        assert_eq!(cpu.get_reg(Register::VF), 0x01);
        
        cpu.set_reg(Register::V0, 0x01);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.get_reg(Register::V0), 0x02);
        assert_eq!(cpu.get_reg(Register::VF), 0x00);
 
    }

    //0x9000
    #[test]
    fn instruction_skip_reg_not_equal() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x9010);
        let start_pc = cpu.pc;
        cpu.set_reg(Register::V0, 0x42);
        cpu.set_reg(Register::V1, 0x84);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, start_pc + 2);
    }
    #[test]
    fn instruction_skip_reg_not_equal_fail() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0x9011);
        cpu.set_reg(Register::V0, 0x42);
        cpu.set_reg(Register::V1, 0x84);
        let res = cpu.execute(&mut mem, &mut display, ins);
        assert!(res.is_err());
    }

    //0x0A000
    #[test]
    fn instruction_ld_idx() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0xAFFF);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.idx, 0x0FFF);
    }

    //0xB000
    #[test]
    fn instruction_jmp_v0() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0xB200);
        cpu.set_reg(Register::V0, 0x42);
        let _ = cpu.execute(&mut mem, &mut display, ins);
        assert_eq!(cpu.pc, 0x0242);
    }

    //0xC000
    fn instruction_reg_rnd() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
        let ins = Instruction(0xC0FF);
        unimplemented!();
    }

    //0xD0000
    #[test]
    fn logo() {
        let mut mem = Memory::new();
        let mut cpu = CPU::new();
        let mut display = Display::new();
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
        mem.load(&data, cpu.pc as usize);
        for i in 0..25 {
            let ins = cpu.fetch(&mem).unwrap();
            cpu.execute(&mut mem, &mut display, ins);
        }
        println!("{}", display);
    }

    //OxE0000
    #[test]
    fn skip_next_instruction_key_pressed() {
        unimplemented!();
    }

    #[test]
    fn skip_next_instruction_key_not_pressed() {
        unimplemented!();
    }

    //0xF000
    #[test]
    fn ld_dt_ref() {
        let mut cpu = CPU::new();
        cpu.delay = 0x42;
        let ins = Instruction(0xF007);
        assert_eq!(cpu.get_reg(Register::V0), 0x42);
    }

    #[test]
    fn wait_for_key_press() {
        unimplemented!();
    }

    #[test]
    fn set_delay_timer() {
        let mut cpu = CPU::new();
        cpu.set_reg(Register::V0, 0x42);
        let ins = Instruction(0xF015);
        assert_eq!(cpu.delay, 0x42);
    }

    #[test]
    fn set_sound_timer() {
        let mut cpu = CPU::new();
        cpu.set_reg(Register::V0, 0x42);
        let ins = Instruction(0xF018);
        assert_eq!(cpu.sound, 0x42);
    }

    #[test]
    fn idx_add_reg() {
        let mut cpu = CPU::new();
        cpu.set_reg(Register::V0, 0x42);
        cpu.idx = 0x0001;
        let ins = Instruction(0xF01E);
        assert_eq!(cpu.idx, 0x0043);
        
    }

    #[test]
    fn set_idx_to_digit() {
        let mut cpu = CPU::new();
        cpu.set_reg(Register::V0, 0x42);
        let ins = Instruction(0xF029);
        unimplemented!();
    }

    #[test]
    fn store_bcd_at_idx() {
        let ins = Instruction(0xF033);
        unimplemented!()
    }

    #[test]
    fn store_all_reg_from_idx() {
        let ins = Instruction(0xF055);
        unimplemented!()
    } 

    #[test]
    fn load_all_reg_from_idx() {
        let ins = Instruction(0xF065);
        unimplemented!()
    } 

}
