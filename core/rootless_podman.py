from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tomllib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.managed_workspace import WorkspaceStage
from shared.workspace_binding import WorkspaceUnavailable

QUALIFIED_PYTHON_IMAGE = "37ccb536022bb36c4570856b55ebfa85bf07827e3c2fa3eba37f0dac62eebc05"
QUALIFIED_PYTHON_DIGEST = "sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487"
QUALIFIED_AMD64_DIGEST = "sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
QUALIFIED_PROFILE_VERSION = "i1-ubuntu-podman-5.7-python-v2"


def _field(value: object, *path: str) -> object:
    for name in path:
        if not isinstance(value, dict):
            raise WorkspaceUnavailable("invalid rootless engine metadata")
        value = value.get(name)
    return value


def _signature(path: Path, *, directory: bool = False) -> tuple[int, ...]:
    info = path.lstat()
    if directory:
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise WorkspaceUnavailable("rootless state directory must be private and owned")
        return info.st_dev, info.st_ino, info.st_mode, info.st_uid
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o022:
        raise WorkspaceUnavailable("rootless profile file must be regular and protected")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _required_field(value: object, *path: str) -> object:
    for name in path:
        if not isinstance(value, dict) or name not in value:
            raise WorkspaceUnavailable("incomplete effective container profile")
        value = value[name]
    return value


@dataclass(frozen=True)
class PreparedPythonContainer:
    container_id: str
    name: str
    stage: WorkspaceStage
    script: str
    profile_revision: str


@dataclass(frozen=True)
class RootlessPodmanProfile:
    """Конкретный проверенный local profile; ни socket, ни generic allow flag.

    Qualify читает фактические engine/image metadata. Перед каждым control call
    проверяются исходные UID, configuration и executable bindings. Никакой
    download/daemon start/config repair внутри runtime не выполняется.
    """

    state_root: Path
    uid: int
    gid: int
    revision: str
    _directories: tuple[tuple[Path, tuple[int, ...]], ...]
    _files: tuple[tuple[Path, tuple[int, ...]], ...]

    @classmethod
    def qualify(cls, state_root: Path) -> RootlessPodmanProfile:
        if sys.platform != "linux":
            raise WorkspaceUnavailable("rootless execution requires a qualified Linux host")
        if os.geteuid() == 0:
            raise WorkspaceUnavailable("rootless execution requires non-root identity")
        state_root = state_root.absolute()
        canonical = state_root.resolve()
        for sandbox in (
            Path(__file__).resolve().parents[1] / "sandbox",
            Path.cwd() / "sandbox",
        ):
            sandbox = sandbox.resolve()
            if canonical.is_relative_to(sandbox) or sandbox.is_relative_to(canonical):
                raise WorkspaceUnavailable("rootless state overlaps ordinary sandbox resources")
        if any(character in str(state_root) for character in (",", ":", "\n", "\r", "\x00")):
            raise WorkspaceUnavailable("unsupported rootless state path")
        try:
            directories = tuple(
                (path, _signature(path, directory=True))
                for path in (
                    state_root,
                    *(
                        state_root / name
                        for name in ("home", "config", "data", "storage", "runroot", "tmp", "hooks")
                    ),
                    Path(f"/run/user/{os.geteuid()}"),
                )
            )
            config = state_root / "containers.conf"
            config_signature = _signature(config)
            if config_signature[3] != os.geteuid() or config_signature[4] > 4096:
                raise WorkspaceUnavailable("unexpected rootless engine configuration")
            fd = os.open(config, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                pinned = os.fstat(fd)
                if (pinned.st_dev, pinned.st_ino) != config_signature[:2]:
                    raise WorkspaceUnavailable("rootless configuration inode changed")
                content = os.read(fd, 4097)
                if _signature(config) != config_signature:
                    raise WorkspaceUnavailable("rootless configuration changed")
            finally:
                os.close(fd)
            if len(content) > 4096 or tomllib.loads(content.decode()) != {
                "engine": {"runtime": "runc"}
            }:
                raise WorkspaceUnavailable("unexpected rootless engine configuration")
            files = tuple(
                (path, _signature(path))
                for path in (
                    config,
                    Path("/usr/bin/podman"),
                    Path("/usr/bin/runc"),
                    Path("/usr/bin/conmon"),
                    Path("/usr/share/containers/seccomp.json"),
                )
            )
            if files[0][1][3] != os.geteuid() or any(
                signature[3] != 0 for _, signature in files[1:]
            ):
                raise WorkspaceUnavailable("rootless profile file owner mismatch")
            revision = hashlib.sha256(
                repr(
                    (QUALIFIED_PROFILE_VERSION, QUALIFIED_PYTHON_IMAGE, directories, files)
                ).encode()
            ).hexdigest()
            profile = cls(state_root, os.geteuid(), os.getegid(), revision, directories, files)
            profile.check_current()
            info = profile.json_control(["info", "--format=json"])
            expected = {
                ("host", "security", "rootless"): True,
                ("host", "security", "seccompEnabled"): True,
                ("host", "security", "seccompProfilePath"): "/usr/share/containers/seccomp.json",
                ("host", "ociRuntime", "name"): "runc",
                ("host", "ociRuntime", "path"): "/usr/bin/runc",
                ("host", "ociRuntime", "package"): "runc_1.4.0-0ubuntu1_amd64",
                ("host", "conmon", "path"): "/usr/bin/conmon",
                ("host", "conmon", "package"): "conmon_2.1.13+ds1-2_amd64",
                ("version", "Version"): "5.7.0",
                ("version", "BuildOrigin"): "Ubuntu",
                ("version", "OsArch"): "linux/amd64",
                ("store", "graphDriverName"): "vfs",
                ("store", "graphRoot"): str(state_root / "storage"),
                ("store", "runRoot"): str(state_root / "runroot"),
            }
            for path, value in expected.items():
                actual = _field(info, *path)
                if type(actual) is not type(value) or actual != value:
                    raise WorkspaceUnavailable("rootless engine differs from qualified profile")
            images = profile.json_control(["image", "inspect", QUALIFIED_PYTHON_IMAGE])
            if not isinstance(images, list) or len(images) != 1:
                raise WorkspaceUnavailable("qualified image unavailable")
            image = images[0]
            for key, value in {
                "Id": QUALIFIED_PYTHON_IMAGE,
                "Os": "linux",
                "Architecture": "amd64",
            }.items():
                if _field(image, key) != value:
                    raise WorkspaceUnavailable("image differs from qualified profile")
            # Podman может выбрать index либо platform manifest в поле Digest.
            # Оба уже проверены I1; immutable image ID и official repository binding обязательны.
            if _field(image, "Digest") not in (QUALIFIED_PYTHON_DIGEST, QUALIFIED_AMD64_DIGEST):
                raise WorkspaceUnavailable("image manifest differs from qualified profile")
            digests = _field(image, "RepoDigests")
            if not isinstance(digests, list) or not all(
                f"docker.io/library/python@{digest}" in digests
                for digest in (QUALIFIED_PYTHON_DIGEST, QUALIFIED_AMD64_DIGEST)
            ):
                raise WorkspaceUnavailable("qualified image repository binding unavailable")
            if _field(image, "Config", "Entrypoint") not in (None, []):
                raise WorkspaceUnavailable("unexpected image entrypoint")
            if _field(image, "Config", "Volumes") not in (None, {}):
                raise WorkspaceUnavailable("unexpected image volumes")
            profile.check_current()
            return profile
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if isinstance(exc, WorkspaceUnavailable):
                raise
            raise WorkspaceUnavailable("rootless profile qualification unavailable") from exc

    def check_current(self) -> None:
        if os.geteuid() != self.uid or os.getegid() != self.gid or self.uid == 0:
            raise WorkspaceUnavailable("rootless identity changed")
        try:
            for path, expected in self._directories:
                if _signature(path, directory=True) != expected:
                    raise WorkspaceUnavailable("rootless state directory binding changed")
            for path, expected in self._files:
                if _signature(path) != expected:
                    raise WorkspaceUnavailable("rootless profile file binding changed")
            if next((self.state_root / "hooks").iterdir(), None) is not None:
                raise WorkspaceUnavailable("rootless hooks are not empty")
            # mounts.conf может добавлять secrets при start, не только caller bind mounts.
            for path in (
                Path("/usr/share/containers/mounts.conf"),
                Path("/etc/containers/mounts.conf"),
                self.state_root / "config/containers/mounts.conf",
                self.state_root / "home/.config/containers/mounts.conf",
            ):
                if os.path.lexists(path):
                    raise WorkspaceUnavailable("automatic container mounts are unqualified")
        except OSError as exc:
            raise WorkspaceUnavailable("rootless profile binding unavailable") from exc

    def environment(self) -> dict[str, str]:
        root = self.state_root
        return {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(root / "home"),
            "XDG_RUNTIME_DIR": f"/run/user/{self.uid}",
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "CONTAINERS_CONF": str(root / "containers.conf"),
            "TMPDIR": str(root / "tmp"),
            "LANG": "C.UTF-8",
        }

    def argv(self) -> list[str]:
        root = self.state_root
        return [
            "/usr/bin/podman",
            "--remote=false",
            "--root",
            str(root / "storage"),
            "--runroot",
            str(root / "runroot"),
            "--storage-driver=vfs",
            "--tmpdir",
            str(root / "tmp"),
            "--events-backend=file",
            "--hooks-dir",
            str(root / "hooks"),
        ]

    def control(
        self, args: list[str], *, timeout: float = 15
    ) -> subprocess.CompletedProcess[bytes]:
        self.check_current()
        return subprocess.run(
            self.argv() + args,
            env=self.environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            close_fds=True,
            timeout=timeout,
            check=False,
        )

    def json_control(self, args: list[str]) -> object:
        result = self.control(args)
        if result.returncode != 0:
            raise WorkspaceUnavailable("rootless engine metadata unavailable")
        try:
            value: object = json.loads(result.stdout)
            return value
        except (ValueError, UnicodeError) as exc:
            raise WorkspaceUnavailable("invalid rootless engine metadata") from exc

    def create_python_container(
        self, stage: WorkspaceStage, script: str
    ) -> PreparedPythonContainer:
        if stage.binding.environment_revision != self.revision:
            raise WorkspaceUnavailable("workspace environment binding changed")
        relative = Path(script)
        if (
            not script
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".py"
            or any(character in str(stage.root) for character in (",", ":", "\n", "\r", "\x00"))
        ):
            raise WorkspaceUnavailable("invalid isolated Python script path")
        stage_signature = _signature(stage.root, directory=True)
        if stage_signature[:2] != stage.identity:
            raise WorkspaceUnavailable("workspace stage identity changed")
        name = f"slavikai-workspace-{uuid.uuid4().hex}"
        command = ["python", "--", f"/workspace/{relative.as_posix()}"]
        result = self.control(
            [
                "create",
                "--name",
                name,
                "--label",
                f"slavikai.workspace.operation={name}",
                "--pull=never",
                "--network=none",
                "--pid=private",
                "--ipc=private",
                "--uts=private",
                "--userns=keep-id",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=32m",
                "--pids-limit=64",
                "--memory=256m",
                "--workdir=/workspace",
                "--env=HOME=/tmp/home",
                "--http-proxy=false",
                "--image-volume=ignore",
                "--mount",
                f"type=bind,src={stage.root},dst=/workspace,rw,nosuid,nodev",
                QUALIFIED_PYTHON_IMAGE,
                *command,
            ]
        )
        if result.returncode != 0:
            raise WorkspaceUnavailable("isolated container creation failed; execution not started")
        container_id = result.stdout.strip().decode("ascii")
        if len(container_id) != 64 or any(
            character not in "0123456789abcdef" for character in container_id
        ):
            raise WorkspaceUnavailable("container identity unavailable; execution not started")
        prepared = PreparedPythonContainer(container_id, name, stage, command[-1], self.revision)
        try:
            self.verify_created(prepared)
            if result.stderr:
                raise WorkspaceUnavailable("unexpected isolated container creation diagnostic")
            return prepared
        except (OSError, ValueError, subprocess.SubprocessError):
            # Только ID, полученный от собственного успешного create; чужие jobs не затрагиваются.
            self.control(["rm", "--force", "--time=1", "--ignore", container_id])
            raise

    def verify_created(self, prepared: PreparedPythonContainer) -> None:
        if (
            prepared.profile_revision != self.revision
            or prepared.stage.binding.environment_revision != self.revision
        ):
            raise WorkspaceUnavailable("container environment binding changed")
        signature = _signature(prepared.stage.root, directory=True)
        if signature[:2] != prepared.stage.identity:
            raise WorkspaceUnavailable("workspace stage identity changed")
        records = self.json_control(["inspect", prepared.container_id])
        if not isinstance(records, list) or len(records) != 1:
            raise WorkspaceUnavailable("effective container metadata unavailable")
        record = records[0]
        expected: dict[tuple[str, ...], object] = {
            ("Id",): prepared.container_id,
            ("Name",): prepared.name,
            ("Image",): QUALIFIED_PYTHON_IMAGE,
            ("OCIRuntime",): "runc",
            ("State", "Status"): "created",
            ("State", "Running"): False,
            ("State", "Pid"): 0,
            ("Config", "WorkingDir"): "/workspace",
            ("Config", "User"): f"{self.uid}:{self.gid}",
            ("Config", "Cmd"): ["python", "--", prepared.script],
            ("Config", "Tty"): False,
            ("Config", "OpenStdin"): False,
            ("Config", "AttachStdin"): False,
            ("Config", "Entrypoint"): None,
            ("Config", "Volumes"): None,
            ("Config", "Labels", "slavikai.workspace.operation"): prepared.name,
            ("HostConfig", "NetworkMode"): "none",
            ("HostConfig", "PortBindings"): {},
            ("HostConfig", "PublishAllPorts"): False,
            ("HostConfig", "Privileged"): False,
            ("HostConfig", "ReadonlyRootfs"): True,
            ("HostConfig", "PidMode"): "private",
            ("HostConfig", "IpcMode"): "private",
            ("HostConfig", "UTSMode"): "private",
            ("HostConfig", "CgroupMode"): "private",
            ("HostConfig", "UsernsMode"): "private",
            ("HostConfig", "Devices"): [],
            ("HostConfig", "CapAdd"): [],
            ("HostConfig", "GroupAdd"): [],
            ("HostConfig", "VolumesFrom"): None,
            ("HostConfig", "Links"): None,
            ("HostConfig", "Memory"): 268435456,
            ("HostConfig", "PidsLimit"): 64,
            ("HostConfig", "SecurityOpt"): ["no-new-privileges"],
            ("HostConfig", "RestartPolicy"): {"Name": "no", "MaximumRetryCount": 0},
            ("HostConfig", "AutoRemove"): False,
            ("HostConfig", "Tmpfs"): {"/tmp": "rw,nosuid,nodev,noexec,size=32m,rprivate,tmpcopyup"},
            ("EffectiveCaps",): None,
            ("BoundingCaps",): None,
        }
        for path, value in expected.items():
            actual = _required_field(record, *path)
            if type(actual) is not type(value) or actual != value:
                raise WorkspaceUnavailable("effective container differs from qualified profile")
        environment = _required_field(record, "Config", "Env")
        expected_environment = {
            "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "container=podman",
            "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
            "PYTHON_VERSION=3.12.16",
            "PYTHON_SHA256=b5796915dc45ff745cfb4e990a7e7c93d6f546328a102ce31448d69ebc37575c",
            "LANG=C.UTF-8",
            "HOME=/tmp/home",
        }
        if (
            not isinstance(environment, list)
            or len(environment) != len(expected_environment)
            or any(
                not isinstance(value, str) or value not in expected_environment
                for value in environment
            )
            or set(environment) != expected_environment
        ):
            raise WorkspaceUnavailable("effective container environment is unqualified")
        mounts = _required_field(record, "Mounts")
        if not isinstance(mounts, list) or len(mounts) != 1:
            raise WorkspaceUnavailable("effective container mounts are unqualified")
        for key, value in {
            "Type": "bind",
            "Source": str(prepared.stage.root),
            "Destination": "/workspace",
            "RW": True,
            "Propagation": "rprivate",
            "Options": ["nosuid", "nodev", "rbind"],
        }.items():
            actual = _required_field(mounts[0], key)
            if type(actual) is not type(value) or actual != value:
                raise WorkspaceUnavailable(
                    "effective container mount differs from qualified profile"
                )
        self.check_current()

    def _owned_record(self, prepared: PreparedPythonContainer) -> object:
        if prepared.profile_revision != self.revision:
            raise WorkspaceUnavailable("container environment binding changed")
        records = self.json_control(["inspect", prepared.container_id])
        if not isinstance(records, list) or len(records) != 1:
            raise WorkspaceUnavailable("owned container metadata unavailable")
        record = records[0]
        for path, expected in {
            ("Id",): prepared.container_id,
            ("Name",): prepared.name,
            ("Config", "Labels", "slavikai.workspace.operation"): prepared.name,
        }.items():
            if _required_field(record, *path) != expected:
                raise WorkspaceUnavailable("container ownership mismatch")
        return record

    def finished_exit(
        self, prepared: PreparedPythonContainer, *, cli_returncode: int
    ) -> int | None:
        if cli_returncode < 0:
            return None
        record = self._owned_record(prepared)
        for path, expected in {
            ("State", "Status"): "exited",
            ("State", "Running"): False,
            ("State", "Pid"): 0,
            ("State", "Error"): "",
        }.items():
            actual = _required_field(record, *path)
            if type(actual) is not type(expected) or actual != expected:
                return None
        code = _required_field(record, "State", "ExitCode")
        if type(code) is not int or not 0 <= code <= 255 or code != cli_returncode:
            return None
        started = _required_field(record, "State", "StartedAt")
        finished = _required_field(record, "State", "FinishedAt")
        if not isinstance(started, str) or not isinstance(finished, str):
            return None
        try:
            start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            if (
                start.tzinfo is None
                or end.tzinfo is None
                or start <= datetime(1970, 1, 1, tzinfo=UTC)
                or end < start
            ):
                return None
        except ValueError:
            return None
        return code

    def remove_owned(self, prepared: PreparedPythonContainer) -> bool:
        self._owned_record(prepared)
        result = self.control(["rm", "--force", "--time=1", "--ignore", prepared.container_id])
        return result.returncode == 0 and not result.stderr
