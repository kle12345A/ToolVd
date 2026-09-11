; Inno Setup script — tao file cai dat (.exe installer) cho AI Movie Studio
; Cach dung:
;   1. Cai Inno Setup: https://jrsoftware.org/isdl.php
;   2. Mo file nay bang Inno Setup -> Build -> Compile
;   3. File cai dat se nam trong thu muc Output\

#define MyAppName "AI Movie Studio"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "AMSR"
#define MyAppExeName "AI Movie Studio.exe"

[Setup]
AppId={{8F2A1C44-AMSR-4B7E-9C12-MOVIESTUDIO01}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=AI_Movie_Studio_Setup_{#MyAppVersion}
SetupIconFile=assets\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "vietnamese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Tao bieu tuong ngoai man hinh"; GroupDescription: "Tuy chon:"

[Files]
; Copy toan bo thu muc build (onedir) vao thu muc cai dat
Source: "dist\AI Movie Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Mo {#MyAppName} ngay"; Flags: nowait postinstall skipifsilent
