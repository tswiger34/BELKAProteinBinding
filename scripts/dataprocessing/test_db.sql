CREATE TABLE TestSMILES (
    ID INT PRIMARY KEY,
    BuildingBlock1_Smiles VARCHAR(200),
    BuildingBlock2_Smiles VARCHAR(200),
    BuildingBlock2_Smiles VARCHAR(200),
    FullMolecule_Smiles VARCHAR(600),
    ProteinTargetID TINYINT,
    Binds BIT
);

CREATE TABLE ProteinTargets (
    ProteinTargetID TINYINT PRIMARY KEY,
    ProteinName VARCHAR(8),
    AF2Confidence REAL,
    AF2Annotations REAL,
    AF2Solvent REAL,
    AF2BindingRegions REAL
);

CREATE TABLE Test2dDescriptors(
    ID INT PRIMARY KEY
);

CREATE TABLE Test3dDescriptors(
    ID INT PRIMARY KEY
);

CREATE TABLE BB1_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE BB2_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE BB3_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE TestFingerprints(
    ID INT PRIMARY KEY
);