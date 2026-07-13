# Optional external-data workflow preserved from the original root entry point.
# See examples/README.md for the required package and data layout.

from std.collections import Dict, List
from read_bin.bin_to_df import DfReader


def main():
    var base_path = "../../quant_models_expl/tmp_data/"

    var bin_paths = Dict[String, List[String]]()
    bin_paths["int32"] = ["date.bin"]
    bin_paths["f64"] = [
        "aus_portret.bin", "aus_portretx.bin", "aut_portret.bin", "aut_portretx.bin", "bel_portret.bin", "bel_portretx.bin",
        "bra_portret.bin", "bra_portretx.bin", "che_portret.bin", "che_portretx.bin", "chl_portret.bin", "chl_portretx.bin",
        "chn_portret.bin", "chn_portretx.bin", "col_portret.bin", "col_portretx.bin", "deu_portret.bin", "deu_portretx.bin",
        "dnk_portret.bin", "dnk_portretx.bin", "egy_portret.bin", "egy_portretx.bin", "esp_portret.bin", "esp_portretx.bin",
        "fin_portret.bin", "fin_portretx.bin", "fra_portret.bin", "fra_portretx.bin", "gbr_portret.bin", "gbr_portretx.bin",
        "grc_portret.bin", "grc_portretx.bin", "hkg_portret.bin", "hkg_portretx.bin", "hun_portret.bin", "hun_portretx.bin",
        "idn_portret.bin", "idn_portretx.bin", "ind_portret.bin", "ind_portretx.bin", "irl_portret.bin", "irl_portretx.bin",
        "ita_portret.bin", "ita_portretx.bin", "jpn_portret.bin", "jpn_portretx.bin", "kor_portret.bin", "kor_portretx.bin",
        "mex_portret.bin", "mex_portretx.bin", "mys_portret.bin", "mys_portretx.bin", "nld_portret.bin", "nld_portretx.bin",
        "nor_portret.bin", "nor_portretx.bin", "nzl_portret.bin", "nzl_portretx.bin", "phl_portret.bin", "phl_portretx.bin",
        "pol_portret.bin", "pol_portretx.bin", "prt_portret.bin", "prt_portretx.bin", "sgp_portret.bin", "sgp_portretx.bin",
        "swe_portret.bin", "swe_portretx.bin", "tha_portret.bin", "tha_portretx.bin", "tur_portret.bin", "tur_portretx.bin",
        "twn_portret.bin", "twn_portretx.bin", "zaf_portret.bin", "zaf_portretx.bin", "ewretd.bin", "ewretx.bin",
        "vwretd.bin", "vwretx.bin", "sprtrn.bin"
    ]
    bin_paths["uint64"] = List[String]()
    bin_paths["uint32"] = List[String]()
    bin_paths["uint8"] = List[String]()
    bin_paths["int64"] = List[String]()

    var rows = 10
    var cols = 5
    var df = DfReader[4096, 82](bin_paths, base_path)
    var tensor = df.create()
    DfReader[4096, 82].preview(tensor, rows, cols)
