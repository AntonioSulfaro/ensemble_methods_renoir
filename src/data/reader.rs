use crate::Instance;
use flate2::read::MultiGzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

#[derive(Debug, Clone)]
enum AttributeEncoding {
    Numerical {
        feature_index: usize,
    },
    Categorical {
        feature_index: usize,
        categories: HashMap<String, usize>,
    },
}

#[derive(Debug, Clone)]
pub struct ArffSchema {
    attributes: Vec<AttributeEncoding>,
    pub n_encoded_features: usize,
}

fn open_reader(path: &str) -> Box<dyn BufRead + Send> {
    let file = File::open(path).expect("cannot open ARFF file");
    if path.ends_with(".gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 – header only → schema
// ─────────────────────────────────────────────────────────────────────────────

fn build_schema(path: &str) -> ArffSchema {
    let mut attribute_lines: Vec<String> = Vec::new();

    for line in open_reader(path).lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        let lower = line.to_lowercase();
        if lower.starts_with("@data") {
            break;
        }
        if !lower.starts_with("@attribute") {
            continue;
        }
        attribute_lines.push(line);
    }

    // Last attribute is always the label — exclude it from feature encoding
    let feature_lines = &attribute_lines[..attribute_lines.len().saturating_sub(1)];

    let mut attributes: Vec<AttributeEncoding> = Vec::new();
    let mut n_encoded = 0usize;

    for line in feature_lines {
        let after_keyword = line["@attribute".len()..].trim();
        let type_str = if after_keyword.starts_with('\'') {
            let end = after_keyword[1..].find('\'').map(|i| i + 2).unwrap_or(1);
            after_keyword[end..].trim()
        } else {
            after_keyword
                .splitn(2, char::is_whitespace)
                .nth(1)
                .unwrap_or("")
                .trim()
        };

        if type_str.starts_with('{') {
            let inner = type_str.trim_start_matches('{').trim_end_matches('}');
            let mut categories = HashMap::new();
            for (offset, cat) in inner.split(',').enumerate() {
                categories.insert(cat.trim().to_string(), offset);
            }
            attributes.push(AttributeEncoding::Categorical {
                feature_index: n_encoded,
                categories,
            });
            n_encoded += 1;
        } else {
            attributes.push(AttributeEncoding::Numerical {
                feature_index: n_encoded,
            });
            n_encoded += 1;
        }
    }

    ArffSchema {
        attributes,
        n_encoded_features: n_encoded,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 – data lines only, no feature vectors → n_instances + n_classes
// ─────────────────────────────────────────────────────────────────────────────

fn count_instances_and_classes(path: &str, n_raw_cols: usize) -> (usize, HashMap<String, usize>) {
    let mut in_data = false;
    let mut n_instances = 0usize;
    let mut label_map: HashMap<String, usize> = HashMap::new();
    let mut next_label_id = 0usize;

    for line in open_reader(path).lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if !in_data {
            if line.to_lowercase().starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let label_str: Option<&str> = if line.starts_with('{') {
            let content = &line[1..line.len() - 1];
            content.split(',').find_map(|part| {
                let mut kv = part.trim().splitn(2, char::is_whitespace);
                let idx: usize = kv.next()?.parse().ok()?;
                if idx == n_raw_cols { kv.next() } else { None }
            })
        } else {
            line.split(',')
                .nth(n_raw_cols)
                .map(str::trim)
                .filter(|s| *s != "?")
        };
        if let Some(s) = label_str {
            label_map.entry(s.to_string()).or_insert_with(|| {
                let id = next_label_id;
                next_label_id += 1;
                id
            });
        }
        n_instances += 1;
    }
    (n_instances, label_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 – streaming iterator
// ─────────────────────────────────────────────────────────────────────────────

pub struct ArffStreamIter {
    reader: Box<dyn BufRead + Send>,
    schema: ArffSchema,
    in_data: bool,
    next_id: usize,
    n_raw_cols: usize,
    label_map: HashMap<String, usize>,
}

impl ArffStreamIter {
    fn new(path: &str, schema: ArffSchema, label_map: HashMap<String, usize>) -> Self {
        let n_raw_cols = schema.attributes.len();
        ArffStreamIter {
            reader: open_reader(path),
            schema,
            in_data: false,
            next_id: 0,
            n_raw_cols,
            label_map,
        }
    }

    fn resolve_label(&self, s: &str) -> Option<usize> {
        if s == "?" {
            None
        } else {
            self.label_map.get(s).copied()
        }
    }

    #[inline]
    fn encode_value(attr: &AttributeEncoding, raw: &str, features: &mut [f64]) {
        match attr {
            AttributeEncoding::Numerical { feature_index } => {
                features[*feature_index] = raw.parse::<f64>().unwrap_or(0.0);
            }
            AttributeEncoding::Categorical {
                feature_index,
                categories,
            } => {
                features[*feature_index] = categories.get(raw).copied().unwrap_or(0) as f64;
            }
        }
    }

    fn parse_dense(&mut self, line: &str) -> Option<(usize, Arc<Instance>)> {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < self.n_raw_cols {
            return None;
        }
        let mut features = vec![0.0f64; self.schema.n_encoded_features];
        for (i, attr) in self.schema.attributes.iter().enumerate() {
            Self::encode_value(attr, parts[i], &mut features);
        }
        let label = parts
            .get(self.n_raw_cols)
            .and_then(|&s| self.resolve_label(s));
        let id = self.next_id;
        self.next_id += 1;

        let features = Arc::from(features);
        Some((id, Arc::new(Instance { features, label })))
    }

    fn parse_sparse(&mut self, line: &str) -> Option<(usize, Arc<Instance>)> {
        let content = &line[1..line.len() - 1];
        let mut features = vec![0.0f64; self.schema.n_encoded_features];
        let mut label: Option<usize> = None;
        for part in content.split(',').map(str::trim) {
            let mut kv = part.splitn(2, char::is_whitespace);
            let idx: usize = match kv.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let val = kv.next().unwrap_or("").trim();
            if idx == self.n_raw_cols {
                label = self.resolve_label(val);
            } else if let Some(attr) = self.schema.attributes.get(idx) {
                let attr = attr.clone();
                Self::encode_value(&attr, val, &mut features);
            }
        }
        let id = self.next_id;
        self.next_id += 1;

        let features = Arc::from(features);
        Some((id, Arc::new(Instance { features, label })))
    }
}

impl Iterator for ArffStreamIter {
    type Item = (usize, Arc<Instance>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None,
                Err(_) => continue,
                Ok(_) => {}
            }
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with('%') {
                continue;
            }
            if !self.in_data {
                let lower = line.to_lowercase();
                if lower.starts_with("@data") {
                    self.in_data = true;
                    continue;
                }
                if !line.starts_with('@') {
                    self.in_data = true;
                    // fall through to parse this line
                } else {
                    continue;
                }
            }
            return if line.starts_with('{') {
                self.parse_sparse(&line)
            } else {
                self.parse_dense(&line)
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `(iter, n_classes, n_features, n_instances)` —
pub fn read_arff(path: &str) -> (ArffStreamIter, usize, usize, usize) {
    let schema = build_schema(path);
    let n_features = schema.attributes.len();
    let n_raw_cols = n_features;
    let (n_instances, label_map) = count_instances_and_classes(path, n_raw_cols);
    let n_classes = label_map.len();
    let iter = ArffStreamIter::new(path, schema, label_map);
    (iter, n_classes, n_features, n_instances)
}
