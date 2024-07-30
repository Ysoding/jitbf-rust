use std::{
    collections::VecDeque,
    fmt::Debug,
    fs::File,
    io::{self, Read},
    process,
};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filepath
    #[arg(short, long)]
    file_path: String,
}

fn main() {
    let args = Args::parse();
    let mut file = File::open(&args.file_path).expect("Failed to open input file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read input file");

    let mut ops = Ops::new();
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
                    eprintln!("{}: ERROR: Unbalanced loop", args.file_path);
                    process::exit(1);
                }

                let addr = stack.pop().unwrap();
                ops.add(Op {
                    kind: OpKind::JumpIfNonZero,
                    operand: addr as i32 + 1,
                });
                ops.items[addr].operand = ops.count() as i32;
                cur_op = lexer.next();
            }
        }
    }
    interpreter(&ops)

    // match jit_compile(&ops) {
    //     Ok(code) => {
    //         let mut memory = vec![0u8; 10 * 1000 * 1000];
    //         code(&mut memory);
    //     }
    //     Err(e) => eprintln!("Error: {}", e),
    // }
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

    pub fn pos(&self) -> usize {
        self.position
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
            OpKind::Input => panic!("not implemented"),
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

type AsmFuncType = extern "C" fn(*mut Vec<u8>);

fn jit_compile(ops: &Ops) -> Result<AsmFuncType, io::Error> {
    let mut sb = Vec::new();

    for op in &ops.items {
        match op.kind {
            OpKind::Inc => sb.extend_from_slice(&[0xFE, 0x07]), // inc byte [rdi]
            _ => panic!("not implemented"),
        }
    }

    sb.push(0xC3); // ret

    let data = sb;
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };

    if addr == libc::MAP_FAILED {
        panic!("{}", io::Error::last_os_error());
    }

    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len());
    }

    //     void (*run)(void *memory);
    let code: AsmFuncType = unsafe { std::mem::transmute(addr) };

    Ok(code)
}
