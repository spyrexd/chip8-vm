use derive_try_from_primitive::TryFromPrimitive;
use pretty_hex::pretty_hex;
use std::{convert::TryFrom, fmt};
use std::ops;

const MEM_SIZE: usize = 4096;

//128 x 64
const DISPLAY_SIZE: (usize, usize) = (128,64);

#[derive(Debug)]
pub enum CpuError {
    InvalidMemoryAccess(u16),
    InvalidRegister(<Register as TryFrom<u8>>::Error),
    InvalidOpCode(<OpCode as TryFrom<u16>>::Error),
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

#[derive(Debug, TryFromPrimitive, PartialEq)]
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

#[derive(Debug, Default)]
pub struct CPU {
    pc: u16,
    idx: u16,
    registers: [u8; 16],
    delay: u8,
    sound: u8,
    stack: Vec<u16>,
}

impl CPU {

    fn new() -> Self {
        CPU {
            pc: 0x0200,
            ..Default::default()
        }
    }

    fn set_reg(&mut self, reg: Register, value: u8) {
        self.registers[reg as usize] = value;
    }

    fn get_reg(&self, reg: Register) -> u8 {
        self.registers[reg as usize]
    }

    fn fetch(&mut self, &ref memory: &Memory) -> Result<Instruction, CpuError> {
        if self.pc >= MEM_SIZE as u16 {
           return Err(CpuError::InvalidMemoryAccess(self.pc));
        }
        let ins = Instruction(((memory[self.pc] as u16) << 8) | memory[self.pc +1] as u16);
        self.pc += 2;
        Ok(ins)
    }
}


#[derive(Debug, PartialEq)]
struct Instruction(u16);

impl Instruction {
    fn opcode(&self) -> Result<OpCode, CpuError> {
        Ok(OpCode::try_from(self.0 & 0xF000)?)
    }

    fn registers(&self) -> Result<(Register, Register), CpuError> {
        let first = (self.0 & 0x0F00) >> 8;
        let second = (self.0 & 0x00F0) >> 4;
        Ok((Register::try_from(first as u8)?, Register::try_from(second as u8)?))

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
        let ins = Instruction(0x0123);
        assert_eq!(ins.opcode().unwrap(), OpCode::OC_0);
        assert_eq!(ins.registers().unwrap(), (Register::V1, Register::V2));

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
}
