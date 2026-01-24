use mefikit::prelude as mf;

pub fn etype_to_str(et: mf::ElementType) -> String {
    use mf::ElementType::*;
    match et {
        VERTEX => "VERTEX",
        SEG2 => "SEG2",
        SEG3 => "SEG3",
        SEG4 => "SEG4",
        SPLINE => "SPLINE",
        TRI3 => "TRI3",
        TRI6 => "TRI6",
        TRI7 => "TRI7",
        QUAD4 => "QUAD4",
        QUAD8 => "QUAD8",
        QUAD9 => "QUAD9",
        PGON => "PGON",
        TET4 => "TET4",
        TET10 => "TET10",
        HEX8 => "HEX8",
        HEX21 => "HEX21",
        PHED => "PHED",
    }
    .to_string()
}

pub fn str_to_etype(et: &str) -> mf::ElementType {
    use mf::ElementType::*;
    match et {
        "VERTEX" => VERTEX,
        "SEG2" => SEG2,
        "SEG3" => SEG3,
        "SEG4" => SEG4,
        "TRI3" => TRI3,
        "TRI6" => TRI6,
        "TRI7" => TRI7,
        "QUAD4" => QUAD4,
        "QUAD8" => QUAD8,
        "QUAD9" => QUAD9,
        "TET4" => TET4,
        "TET10" => TET10,
        "HEX8" => HEX8,
        "HEX21" => HEX21,
        _ => panic!("Unsupported element type: {}", et),
    }
}
