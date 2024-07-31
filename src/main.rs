use std::{
    collections::VecDeque,
    fmt::Debug,
    fs::File,
    io::{self, Read},
    process,
};

use clap::Parser;

const JIT_MEMORY_CAP: usize = 10 * 1000 * 1000;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filepath
    #[arg(short, long)]
    file_path: String,

    #[arg(long, default_value_t = false)]
    no_jit: bool,
}

fn main() {
    let args = Args::parse();
    let mut ops = Ops::new();

    if let Err(e) = generate_ops(&args.file_path, &mut ops) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    // let mut i = 0;
    // for op in &ops.items {
    //     println!("{} {:?} {}", i, op.kind, op.operand);
    //     i += 1;
    // }

    if args.no_jit {
        interpreter(&ops)
    } else {
        match jit_compile(&ops) {
            Ok(code) => {
                // let mut memory = vec![0u8; JIT_MEMORY_CAP];
                // let mut memory: [u8; JIT_MEMORY_CAP] = [0; JIT_MEMORY_CAP];
                // let mut memory: Box<[u8; JIT_MEMORY_CAP]> = Box::new([0; JIT_MEMORY_CAP]);
                let mut memory: Box<[u8]> = vec![0; JIT_MEMORY_CAP].into_boxed_slice();
                code(memory.as_mut_ptr());
                // println!("{}", memory[0]);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}

fn generate_ops(file_path: &str, ops: &mut Ops) -> Result<(), io::Error> {
    let mut file = File::open(&file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut stack = AddrStack::new();
    let mut lexer = Lexer::new(&contents);

    let mut cur_op = lexer.next();
    loop {
        match cur_op {
            OpKind::EOF => break,
            OpKind::Inc
            | OpKind::Dec
            | OpKind::Left
            | OpKind::Right
            | OpKind::Input
            | OpKind::Output => {
                let mut count = 1;
                let mut next_op = lexer.next();
                while next_op == cur_op {
                    count += 1;
                    next_op = lexer.next();
                }
                ops.add(Op {
                    kind: cur_op,
                    operand: count,
                });
                cur_op = next_op;
            }
            OpKind::JumpIfZero => {
                let addr = ops.count();
                ops.add(Op {
                    kind: cur_op,
                    operand: 0,
                });
                stack.push(addr);
                cur_op = lexer.next();
            }
            OpKind::JumpIfNonZero => {
                if stack.is_empty() {
                    eprintln!("{} {}: ERROR: Unbalanced loop", file_path, lexer.position);
                    process::exit(1);
                }

                let addr = stack.pop().unwrap();
                ops.add(Op {
                    kind: cur_op,
                    operand: addr as i32 + 1,
                });
                ops.items[addr].operand = ops.count() as i32;
                cur_op = lexer.next();
            }
        }
    }
    if !stack.is_empty() {
        eprintln!("{} {}: ERROR: Unbalanced loop", file_path, lexer.position);
        process::exit(1);
    }
    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OpKind {
    EOF,
    Inc,
    Dec,
    Left,
    Right,
    Input,
    Output,
    JumpIfZero,
    JumpIfNonZero,
}

#[derive(Debug, Clone)]
struct Op {
    kind: OpKind,
    operand: i32,
}

struct Ops {
    items: Vec<Op>,
}

impl Ops {
    fn new() -> Self {
        Ops { items: Vec::new() }
    }

    fn add(&mut self, op: Op) {
        self.items.push(op)
    }

    fn count(&self) -> usize {
        self.items.len()
    }
}

struct AddrStack {
    items: VecDeque<usize>,
}

impl AddrStack {
    fn new() -> Self {
        AddrStack {
            items: VecDeque::new(),
        }
    }

    fn push(&mut self, addr: usize) {
        self.items.push_back(addr);
    }

    fn pop(&mut self) -> Option<usize> {
        self.items.pop_back()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

struct Memory {
    items: Vec<u8>,
}

impl Memory {
    fn new() -> Self {
        Memory { items: Vec::new() }
    }

    fn add(&mut self, b: u8) {
        self.items.push(b);
    }

    fn count(&self) -> usize {
        self.items.len()
    }
}

struct Lexer {
    input: Vec<char>,
    position: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            position: 0,
        }
    }

    pub fn next(&mut self) -> OpKind {
        while self.position < self.input.len() && !self.is_bf_cmd(self.input[self.position]) {
            self.position += 1;
        }

        if self.position >= self.input.len() {
            return OpKind::EOF;
        }

        let res = match self.input[self.position] {
            '+' => OpKind::Inc,
            '-' => OpKind::Dec,
            '<' => OpKind::Left,
            '>' => OpKind::Right,
            ',' => OpKind::Input,
            '.' => OpKind::Output,
            '[' => OpKind::JumpIfZero,
            ']' => OpKind::JumpIfNonZero,
            _ => OpKind::EOF,
        };
        self.position += 1;
        res
    }

    fn is_bf_cmd(&self, ch: char) -> bool {
        "+-<>,.[]".contains(ch)
    }
}

fn interpreter(ops: &Ops) {
    let mut memory = Memory::new();
    memory.add(0);
    let mut head = 0;
    let mut ip = 0;

    while ip < ops.count() {
        let op = &ops.items[ip];
        match op.kind {
            OpKind::EOF => break,
            OpKind::Inc => {
                memory.items[head] = memory.items[head].wrapping_add(op.operand as u8);
                ip += 1;
            }
            OpKind::Dec => {
                memory.items[head] = memory.items[head].wrapping_sub(op.operand as u8);
                ip += 1;
            }
            OpKind::Left => {
                if head < op.operand as usize {
                    panic!("RUNTIME ERROR: Memory underflow");
                }
                head -= op.operand as usize;
                ip += 1;
            }
            OpKind::Right => {
                head += op.operand as usize;
                while head >= memory.count() {
                    memory.add(0);
                }
                ip += 1;
            }
            OpKind::Input => {
                let mut buffer = [0; 1];
                io::stdin()
                    .read_exact(&mut buffer)
                    .expect("Failed to read input");
                memory.items[head] = buffer[0];
                ip += 1;
            }
            OpKind::Output => {
                for _ in 0..op.operand {
                    print!("{}", memory.items[head] as char);
                }
                ip += 1;
            }
            OpKind::JumpIfZero => {
                if memory.items[head] == 0 {
                    ip = op.operand as usize;
                } else {
                    ip += 1;
                }
            }
            OpKind::JumpIfNonZero => {
                if memory.items[head] != 0 {
                    ip = op.operand as usize;
                } else {
                    ip += 1;
                }
            }
        }
    }
}

// stack
// type AsmFuncType = extern "C" fn(*mut [u8]);
type AsmFuncType = extern "C" fn(*mut u8);

struct Backpatch {
    operand_byte_addr: u32,
    src_byte_addr: u32,
    dst_op_index: u32,
}

fn jit_compile(ops: &Ops) -> Result<AsmFuncType, io::Error> {
    let mut sb: Vec<u8> = Vec::new();
    let mut backpatches: Vec<Backpatch> = Vec::new();
    let mut addrs = AddrStack::new();

    for op in &ops.items {
        addrs.push(sb.len());
        match op.kind {
            OpKind::Inc => {
                if op.operand >= 256 {
                    panic!("TODO: suuport bigger operands");
                }
                sb.extend_from_slice(b"\x80\x07"); // add byte[rdi],
                sb.push((op.operand & 0xFF) as u8);
            }
            OpKind::Dec => {
                if op.operand >= 256 {
                    panic!("TODO: suuport bigger operands");
                }
                sb.extend_from_slice(b"\x80\x2f"); // sub byte[rdi],
                sb.push((op.operand & 0xFF) as u8);
            }
            OpKind::Left => {
                // TODO: range cheks for OP_LEFT and OP_RIGHT
                sb.extend_from_slice(b"\x48\x81\xef"); // sub rdi,
                sb.extend_from_slice(&op.operand.to_le_bytes());
            }
            OpKind::Right => {
                sb.extend_from_slice(b"\x48\x81\xc7"); // add rdi,
                sb.extend_from_slice(&op.operand.to_le_bytes());
            }
            OpKind::Output => {
                for _ in 0..op.operand {
                    sb.extend_from_slice(b"\x57"); // push rdi
                    sb.extend_from_slice(b"\x48\xc7\xc0\x01\x00\x00\x00"); // mov rax, 1
                    sb.extend_from_slice(b"\x48\x89\xfe"); // mov rsi, rdi
                    sb.extend_from_slice(b"\x48\xc7\xc7\x01\x00\x00\x00"); // mov rdi, 1
                    sb.extend_from_slice(b"\x48\xc7\xc2\x01\x00\x00\x00"); // mov rdx, 1
                    sb.extend_from_slice(b"\x0f\x05"); // syscall
                    sb.extend_from_slice(b"\x5f"); // pop rdi
                }
            }
            OpKind::Input => {
                for _ in 0..op.operand {
                    sb.extend_from_slice(b"\x57"); // push rdi
                    sb.extend_from_slice(b"\x48\xc7\xc0\x00\x00\x00\x00"); // mov rax, 0
                    sb.extend_from_slice(b"\x48\x89\xfe"); // mov rsi, rdi
                    sb.extend_from_slice(b"\x48\xc7\xc7\x00\x00\x00\x00"); // mov rdi, 0
                    sb.extend_from_slice(b"\x48\xc7\xc2\x01\x00\x00\x00"); // mov rdx, 1
                    sb.extend_from_slice(b"\x0f\x05"); // syscall
                    sb.extend_from_slice(b"\x5f"); // pop rdi
                }
            }
            OpKind::JumpIfZero => {
                sb.extend_from_slice(b"\x8a\x07"); // mov al, byte [rdi]
                sb.extend_from_slice(b"\x84\xc0"); // test al, al
                sb.extend_from_slice(b"\x0f\x84"); // jz
                let operand_byte_addr = sb.len();
                // backpatching to addr
                sb.extend_from_slice(b"\x00\x00\x00\x00"); // nop nop nop nop
                let src_byte_addr = sb.len();

                backpatches.push(Backpatch {
                    operand_byte_addr: operand_byte_addr as u32,
                    src_byte_addr: src_byte_addr as u32,
                    dst_op_index: op.operand as u32,
                })
            }
            OpKind::JumpIfNonZero => {
                sb.extend_from_slice(b"\x8a\x07"); // mov al, byte [rdi]
                sb.extend_from_slice(b"\x84\xc0"); // test al, al
                sb.extend_from_slice(b"\x0f\x85"); // jnz
                let operand_byte_addr = sb.len();
                sb.extend_from_slice(b"\x00\x00\x00\x00"); // nop nop nop nop
                let src_byte_addr = sb.len();

                backpatches.push(Backpatch {
                    operand_byte_addr: operand_byte_addr as u32,
                    src_byte_addr: src_byte_addr as u32,
                    dst_op_index: op.operand as u32,
                })
            }
            _ => panic!("Unreachable"),
        }
    }
    addrs.push(sb.len());

    for bp in &backpatches {
        let src_addr = bp.src_byte_addr as i32;
        let dst_addr = addrs.items[bp.dst_op_index as usize] as i32;
        let operand = dst_addr - src_addr;

        let operand_bytes = operand.to_le_bytes();
        let start = bp.operand_byte_addr as usize;
        let end = start + operand_bytes.len();
        sb[start..end].copy_from_slice(&operand_bytes);
    }

    sb.push(0xC3); // ret

    let asm_code_addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            sb.len(),
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };

    if asm_code_addr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }

    unsafe {
        std::ptr::copy_nonoverlapping(sb.as_ptr(), asm_code_addr as *mut u8, sb.len());
    }

    let code: AsmFuncType = unsafe { std::mem::transmute(asm_code_addr) };

    Ok(code)
}
