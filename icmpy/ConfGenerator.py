import os
import yaml
import logging
import tempfile
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from rdkit.Chem import AllChem
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm


# This is necessary to preserve all, including custom e.g. Energy, properties
# while using multiprocessing with RDKit molecules, since it would rely on pickling
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def is_yaml_file(file_path):
    # Check if the file has a .yaml or .yml extension
    if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
        return False

    # Try loading the file to confirm it’s valid YAML
    try:
        with open(file_path, "r") as file:
            yaml.safe_load(file)
        return True
    except yaml.YAMLError as e:
        raise e
    except FileNotFoundError as e:
        raise e


@dataclass
class ConformerResult:
    smiles: str
    mol: Optional[Chem.Mol]
    error: Optional[str]
    success: bool


def save_results(
    results: List[ConformerResult],
    success_file: str = "successful_conformers.sdf",
    error_file: str = "failed_smiles.txt",
):
    """
    Save successful conformers to SDF file and failed SMILES to text file
    """
    # Save successful conformers
    with Chem.SDWriter(success_file) as writer:
        for result in results:
            if result.success:
                writer.write(result.mol)

    # Save failed SMILES
    with open(error_file, "w") as f:
        for result in results:
            if not result.success:
                f.write(f"{result.smiles}\t{result.error}\n")


class ConfGenerator:
    def __init__(
        self,
        config_file: str,
        icm_path: str,
        log_file: str = "confgen.log",
    ):
        """
        Initialize the conformer generator with configuration
        """
        icm_path_ob = Path(icm_path)
        if not icm_path_ob.exists():
            raise ValueError("Path to icm executable does not exist")
        if not icm_path_ob.is_file():
            raise ValueError("Path to icm executable is not a file")

        if os.environ.get("LM_LICENSE_FILE") is None:
            raise ValueError("`LM_LICENSE_FILE` environment variable is not set")

        if not is_yaml_file(config_file):
            raise ValueError("Config file is not a valid YAML file")

        self.icm_path = icm_path
        self.logger = self._setup_logging(log_file)
        self.config = self._load_config(config_file)
        self.base_cmd = self._build_base_command()

    def _setup_logging(self, log_file: str) -> logging.Logger:
        """Sets up logging configuration"""
        logger = logging.getLogger("confgen")
        logger.setLevel(logging.INFO)

        # Check if logger already has handlers to avoid duplicates
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _load_config(self, config_file: str) -> Dict:
        """Loads configuration from YAML file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def _build_base_command(self) -> List[str]:
        """
        Builds the base command list that will be reused for all SMILES
        Placeholder strings will be replaced with actual file paths
        """
        cmd = [self.icm_path, "_confGen", "{input_file}", "{output_file}"]

        # Add numeric parameters
        numeric_params = [
            "auto",
            "effort",
            "maxenergy",
            "mnconf",
            "sizelimit",
            "torlimit",
            "vicinity",
            "diel",
        ]
        for param in numeric_params:
            if param in self.config:
                cmd.append(f"{param}={self.config[param]}")

        # Add flag parameters
        flag_params = {
            "cartesian_mmff": "-c",
            "no_bond_lengths": "-b",
            "formal_charges": "-C",
            "sample_double_bonds": "-d",
            "conf_entropy_only": "-e",
            "force_overwrite": "-f",
            "force_input_update": "-I",
            "quiet": "-q",
            "sample_ring_systems": "-r",
            "systematic_search": "-s",
            "ai_assisted": "-A",
            "evaluate_strain": "-S",
            "verbose_commands": "-V",
            "verbose_info": "-v",
        }

        for param, flag in flag_params.items():
            if self.config.get(param, False):
                cmd.append(flag)

        # Add other parameters
        if self.config.get("keep_hydrogens", False):
            cmd.append("-hydrogen")
        if self.config.get("keep_3d", False):
            cmd.append("-keep3D")
        if "molcart" in self.config:
            cmd.append(f"-molcart={self.config['molcart']}")
        return cmd

    def _sdf_to_smiles(self, sdf_file_path: str) -> Optional[str]:
        """Convert SDF file to SMILES string"""
        suppl = Chem.SDMolSupplier(sdf_file_path)

        # Get the first molecule
        first_molecule = next(suppl)
        if first_molecule is not None:
            try:
                # Convert to SMILES
                smiles = Chem.MolToSmiles(first_molecule)
                return smiles
            except Exception as e:
                error_msg = f"Error converting SDF to SMILES: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            raise ValueError("No molecules found in SDF file")

    def _smiles_to_sdf_file(
        self, smiles: str, temp_dir: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """Convert SMILES to a temporary SDF file"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles_msg = f"Invalid SMILES string: {smiles}"
                self.logger.error(invalid_smiles_msg)
                return "", invalid_smiles_msg

            mol = Chem.AddHs(mol)
            success = AllChem.EmbedMolecule(mol, randomSeed=42)

            if success == -1:
                coords_gen_error_msg = "Failed to generate initial 3D coordinates"
                self.logger.error(coords_gen_error_msg)
                return "", coords_gen_error_msg

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".sdf", delete=False, dir=temp_dir
            )
            temp_file_path = temp_file.name

            writer = Chem.SDWriter(temp_file_path)
            writer.write(mol)
            writer.close()

            return temp_file_path, None

        except Exception as e:
            error_msg = f"Error converting SMILES to SDF: {str(e)}"
            self.logger.error(error_msg)
            return "", f"{error_msg}"

    def _process_single_mol(self, mol_identifier: str) -> ConformerResult:
        """Process a single mol where the input is a path to an SDF file or a SMILES string"""
        temp_input_path = ""
        temp_output_path = ""
        smiles = ""
        smiles_input = None

        try:
            # Create temporary files
            if os.path.isfile(mol_identifier) and mol_identifier.endswith(".sdf"):
                input_path = mol_identifier
                smiles_input = False
                smiles = self._sdf_to_smiles(input_path)
            else:
                smiles_input = True
                smiles = mol_identifier
                input_path, error = self._smiles_to_sdf_file(smiles)
                if error:
                    return ConformerResult(
                        smiles=smiles, mol=None, error=error, success=False
                    )

            temp_input_path, error = self._smiles_to_sdf_file(smiles)
            if error:
                return ConformerResult(
                    smiles=smiles, mol=None, error=error, success=False
                )

            with tempfile.NamedTemporaryFile(
                suffix=".sdf", delete=False
            ) as temp_output:
                temp_output_path = temp_output.name

            # Build command with actual file paths
            cmd = [
                part.format(input_file=temp_input_path, output_file=temp_output_path)
                for part in self.base_cmd
            ]

            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = (
                    f"Command failed with return code {process.returncode}: {stderr}"
                )
                self.logger.error(error_msg)
                return ConformerResult(
                    smiles=smiles, mol=None, error=error_msg, success=False
                )

            sd_supplier = Chem.SDMolSupplier(temp_output_path)
            mol = None

            for i, conformer in enumerate(sd_supplier):
                if conformer is None:
                    continue
                if mol is None:
                    mol = Chem.Mol(conformer)
                    mol.RemoveAllConformers()
                energy = conformer.GetDoubleProp("ENERGY")
                conf_id = mol.AddConformer(conformer.GetConformer(), assignId=True)
                mol.GetConformer(conf_id).SetDoubleProp("ENERGY", energy)

            # Clean up temporary files
            if smiles_input:
                os.unlink(temp_input_path)
            os.unlink(temp_output_path)

            if mol is None:
                error_msg = "Failed to read output conformer"
                self.logger.error(error_msg)
                return ConformerResult(
                    smiles=smiles,
                    mol=None,
                    error=error_msg,
                    success=False,
                )
            else:
                self.logger.info(f"generated conformer for {smiles}")

            return ConformerResult(smiles=smiles, mol=mol, error=None, success=True)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            if os.path.exists(temp_input_path) and smiles_input:
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return ConformerResult(
                smiles=smiles, mol=None, error=error_msg, success=False
            )

    def process_mol_list(
        self, mol_identifiers: List[str], n_jobs: int = 1, show_progress: bool = True
    ) -> List[ConformerResult]:
        """
        Process a list of molecules in parallel

        Args:
            smiles_list: List of SMILES strings to process
            n_jobs: Number of parallel jobs (default: 1)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of ConformerResult objects containing results and any errors
        """
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_mol, mol_identifier)
                for mol_identifier in mol_identifiers
            ]

            if show_progress:
                futures = tqdm(
                    as_completed(futures),
                    total=len(mol_identifiers),
                    desc="Generating conformers",
                )
            else:
                futures = as_completed(futures)

            for future in futures:
                results.append(future.result())

        return results