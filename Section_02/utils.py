import allel, os
import numpy as np

def read_data(vcf: str, ref_ind_file: str,
              tgt_ind_file: str) -> tuple[dict, list, dict, list]:
    """
    Description:
        Helper function for reading data from the reference
        and target populations.

    Arguments:
        vcf str: Name of the VCF file containing genotype data
                 from the reference and target populations.
        ref_ind_file str: Name of the file containing sample information
                          from the reference population.
        tgt_ind_file str: Name of the file containing sample information
                          from the target population.

    Returns:
        ref_data dict: Genotype data from the reference population.
        ref_samples list: Sample information from the reference population.
        tgt_data dict: Genotype data from the target population.
        tgt_samples list: Sample information from the target population.
    """

    ref_data = ref_samples = tgt_data = tgt_samples = None
    ref_samples = parse_ind_file(ref_ind_file)
    ref_data = read_geno_data(vcf, ref_samples)
    tgt_samples = parse_ind_file(tgt_ind_file)
    tgt_data = read_geno_data(vcf, tgt_samples)

    chr_names = tgt_data.keys()
    for c in chr_names:
        # Remove variants fixed in both the reference and target individuals
        ref_fixed_variants = np.sum(
            ref_data[c]['GT'].is_hom_alt(),axis=1
        ) == len(ref_samples)
        tgt_fixed_variants = np.sum(
            tgt_data[c]['GT'].is_hom_alt(),axis=1
        ) == len(tgt_samples)
        fixed_index = np.logical_and(ref_fixed_variants, tgt_fixed_variants)
        index = np.logical_not(fixed_index)
        fixed_pos =ref_data[c]['POS'][fixed_index]
        ref_data = filter_data(ref_data, c, index)
        tgt_data = filter_data(tgt_data, c, index)

    for c in chr_names:
        mut_num, ind_num, ploidy = ref_data[c]['GT'].shape
        ref_data[c]['GT'] = np.reshape(
            ref_data[c]['GT'].values, (mut_num, ind_num * ploidy)
        )
        mut_num, ind_num, ploidy = tgt_data[c]['GT'].shape
        tgt_data[c]['GT'] = np.reshape(
            tgt_data[c]['GT'].values, (mut_num, ind_num * ploidy)
        )

    return ref_data, ref_samples, tgt_data, tgt_samples


def parse_ind_file(filename: str) -> list:
    """
    Description:
        Helper function to read sample information from files.

    Arguments:
        filename str: Name of the file containing sample information.

    Returns:
        samples list: Sample information.
    """

    f = open(filename, 'r')
    samples = [l.rstrip() for l in f.readlines()]
    f.close()

    if len(samples) == 0:
        raise Exception(
            f'No sample is found in {filename}! Please check your data.'
        )

    return samples


def read_geno_data(vcf: str, ind: list) -> dict:
    """
    Description:
        Helper function to read genotype data from VCF files.

    Arguments:
        vcf str: Name of the VCF file containing genotype data.
        ind list: List containing names of samples.

    Returns:
        data dict: Genotype data.
    """
    
    vcf = allel.read_vcf(vcf, alt_number=1, samples=ind)
    gt = vcf['calldata/GT']
    chr_names = np.unique(vcf['variants/CHROM'])
    samples = vcf['samples']
    pos = vcf['variants/POS']
    ref = vcf['variants/REF']
    alt = vcf['variants/ALT']

    data = dict()
    for c in chr_names:
        if c not in data.keys():
            data[c] = dict()
            data[c]['POS'] = pos
            data[c]['REF'] = ref
            data[c]['ALT'] = alt
            data[c]['GT'] = gt
        index = np.where(vcf['variants/CHROM'] == c)
        data = filter_data(data, c, index)

    return data


def filter_data(data: dict, c: str, index: np.ndarray) -> dict:
    """
    Description:
        Helper function to filter genotype data.

    Arguments:
        data dict: Genotype data for filtering.
        c str: Names of chromosomes.
        index numpy.ndarray: A boolean array determines variants to
                             be removed.

    Returns:
        data dict: Genotype data after filtering.
    """

    data[c]['POS'] = data[c]['POS'][index]
    data[c]['REF'] = data[c]['REF'][index]
    data[c]['ALT'] = data[c]['ALT'][index]
    data[c]['GT'] = allel.GenotypeArray(data[c]['GT'][index])

    return data


def create_windows(pos: np.ndarray, chr_name: str,
                   win_step: int, win_len: int) -> list:
    """
    Description:
        Creates sliding windows along the genome.

    Arguments:
        pos numpy.ndarray: Positions for the variants.
        chr_name str: Name of the chromosome.
        win_step int: Step size of sliding windows.
        win_len int: Length of sliding windows.

    Returns:
        windows list: List of sliding windows along the genome.
    """
    win_start = (pos[0]+win_step)//win_step*win_step-win_len
    if win_start < 0: win_start = 0
    last_pos = pos[-1]

    windows = []
    while last_pos > win_start:
        win_end = win_start + win_len
        windows.append((chr_name, win_start, win_end))
        win_start += win_step

    return windows
    
    
def output(res: dict, tgt_samples: np.ndarray, ploidy: int,
           output_dir: str, output_prefix: str) -> None:
    """
    Description:
        Outputs feature vectors.

    Arguments:
        res dict: Feature vectors to be output.
        tgt_samples numpy.ndarray: Name of samples from the target
                                   population.
        ploidy int: Ploidy of the genome.
        output_dir str: Name of the output directory.
        output_prefix str: Prefix of the output filename.
    
    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)

    header = "chrom\tstart\tend\tsample"
    header += "\ttotal_SNP_num"
    header += "\tprivate_SNP_num"
    header += "\tmin_ref_dist"
    header += "\tmax_ref_dist"
    header += "\tmean_ref_dist"
    header += "\tmedian_ref_dist"
    header += "\tvar_ref_dist"
    header += "\tskew_ref_dist"
    header += "\tkurtosis_ref_dist"

    header += "\tmin_tgt_dist"
    header += "\tmax_tgt_dist"
    header += "\tmean_tgt_dist"
    header += "\tmedian_tgt_dist"
    header += "\tvar_tgt_dist"
    header += "\tskew_tgt_dist"
    header += "\tkurtosis_tgt_dist"

    output_file = f'{output_dir}/{output_prefix}.features'
    with open(output_file, 'w') as f:
        f.write(f'{header}\n')
        for r in res:
            chrom = r[0]
            start = r[1]
            end = r[2]
            items = r[3]
            for i in range(len(tgt_samples)*ploidy):
                if ploidy != 1:
                    sample = f'{tgt_samples[int(i/ploidy)]}_{i%ploidy+1}'
                else: sample = tgt_samples[i]
                out = []
                out.append(f'{items["ttl_mut_nums"][i]}')
                out.append(f'{items["pvt_mut_nums"][i]}')
                out.append(f'{items["min_ref_dists"][i]}')
                out.append(f'{items["max_ref_dists"][i]}')
                out.append(f'{items["mean_ref_dists"][i]}')
                out.append(f'{items["median_ref_dists"][i]}')
                out.append(f'{items["var_ref_dists"][i]}')
                out.append(f'{items["skew_ref_dists"][i]}')
                out.append(f'{items["kurtosis_ref_dists"][i]}')

                out.append(f'{items["min_tgt_dists"][i]}')
                out.append(f'{items["max_tgt_dists"][i]}')
                out.append(f'{items["mean_tgt_dists"][i]}')
                out.append(f'{items["median_tgt_dists"][i]}')
                out.append(f'{items["var_tgt_dists"][i]}')
                out.append(f'{items["skew_tgt_dists"][i]}')
                out.append(f'{items["kurtosis_tgt_dists"][i]}')

                out = "\t".join(out)
                f.write(f'{chrom}\t{start}\t{end}\t{sample}\t{out}\n')