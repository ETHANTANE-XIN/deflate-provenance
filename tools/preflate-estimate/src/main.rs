//! preflate-estimate: the preflate-rs baseline used by dfp (proposal III.C).
//!
//! Usage: preflate-estimate FILE...
//! Each FILE holds one raw DEFLATE stream.  One tab-separated line is printed
//! per file:
//!   FILE  OK   <compressed bytes>  <correction bytes>  <TokenPredictorParameters>
//!   FILE  ERR  <message>
//! The parameters are preflate-rs's estimate of the original encoder's
//! settings, including the hash algorithm it detected.

use preflate_rs::{preflate_whole_deflate_stream, PreflateConfig};
use std::io::Write;

fn main() {
    let cfg = PreflateConfig {
        verify_compression: false,
        ..PreflateConfig::default()
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for path in std::env::args().skip(1) {
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                writeln!(out, "{}\tERR\t{}", path, e).unwrap();
                continue;
            }
        };
        match preflate_whole_deflate_stream(&data, &cfg) {
            Ok((r, _plain_text)) => {
                let params = match r.parameters {
                    Some(p) => format!("{:?}", p),
                    None => "None".to_string(),
                };
                writeln!(
                    out,
                    "{}\tOK\t{}\t{}\t{}",
                    path,
                    r.compressed_size,
                    r.corrections.len(),
                    params
                )
                .unwrap();
            }
            Err(e) => {
                let msg = format!("{:?}", e).replace(['\t', '\n'], " ");
                writeln!(out, "{}\tERR\t{}", path, msg).unwrap();
            }
        }
    }
}
