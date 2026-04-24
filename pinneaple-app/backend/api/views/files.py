import os
import struct
import csv
import io
import numpy as np
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from api.models import UploadedFile, Project
from api.serializers import UploadedFileSerializer


class FileUploadView(APIView):
    parser_classes     = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_obj   = request.FILES.get("file")
        project_id = request.data.get("project_id")
        if not file_obj:
            return Response({"error": "no file"}, status=400)

        name      = file_obj.name
        ext       = os.path.splitext(name)[1].lower()
        file_type = {"stl": "stl", ".stl": "stl",
                     ".csv": "csv", ".npy": "npy"}.get(ext, "unknown")
        if ext == ".stl":   file_type = "stl"
        elif ext == ".csv": file_type = "csv"
        elif ext == ".npy": file_type = "npy"

        project = None
        if project_id:
            try:
                project = Project.objects.get(id=project_id, owner=request.user)
            except Project.DoesNotExist:
                pass

        meta = {}
        raw  = file_obj.read()

        if file_type == "stl":
            meta = _parse_stl_meta(raw)
        elif file_type == "csv":
            meta = _parse_csv_meta(raw)
        elif file_type == "npy":
            try:
                arr  = np.load(io.BytesIO(raw))
                meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            except Exception as e:
                meta = {"error": str(e)}

        # Save file (reset position)
        file_obj.seek(0)
        uploaded = UploadedFile.objects.create(
            project   = project,
            name      = name,
            file      = file_obj,
            file_type = file_type,
            meta      = meta,
        )
        return Response(UploadedFileSerializer(uploaded).data, status=201)


class FileListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        project_id = request.query_params.get("project_id")
        qs = UploadedFile.objects.filter(project__owner=request.user).order_by("-created_at")
        if project_id:
            qs = qs.filter(project_id=project_id)
        return Response(UploadedFileSerializer(qs[:100], many=True).data)


class FileDetailView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, file_id):
        try:
            f = UploadedFile.objects.get(id=file_id)
        except UploadedFile.DoesNotExist:
            return Response({"error": "not found"}, status=404)
        return Response(UploadedFileSerializer(f).data)

    def delete(self, request, file_id):
        try:
            f = UploadedFile.objects.get(id=file_id)
        except UploadedFile.DoesNotExist:
            return Response({"error": "not found"}, status=404)
        if f.file and os.path.isfile(f.file.path):
            os.remove(f.file.path)
        f.delete()
        return Response(status=204)


class FileDataView(APIView):
    """Return parsed data from a CSV file as lists."""
    permission_classes = [IsAuthenticated]

    def get(self, request, file_id):
        try:
            f = UploadedFile.objects.get(id=file_id)
        except UploadedFile.DoesNotExist:
            return Response({"error": "not found"}, status=404)

        if f.file_type != "csv":
            return Response({"error": "only CSV supported"}, status=400)

        raw  = f.file.read()
        data = _parse_csv_data(raw)
        return Response(data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_stl_meta(raw: bytes) -> dict:
    try:
        text = raw.decode("ascii")
        if text.strip().startswith("solid"):
            lines   = text.splitlines()
            n_tri   = sum(1 for l in lines if l.strip().startswith("facet normal"))
            n_verts = n_tri * 3
        else:
            n_tri   = struct.unpack_from("<I", raw, 80)[0]
            n_verts = n_tri * 3
        return {"n_triangles": n_tri, "n_vertices": n_verts, "format": "ascii" if text.strip().startswith("solid") else "binary"}
    except Exception as e:
        return {"error": str(e)}


def _parse_stl_vertices(raw: bytes) -> list:
    try:
        text = raw.decode("ascii")
        is_ascii = text.strip().startswith("solid")
    except Exception:
        is_ascii = False

    verts = []
    if is_ascii:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("vertex"):
                verts.append(list(map(float, line.split()[1:4])))
    else:
        n_tri = struct.unpack_from("<I", raw, 80)[0]
        for i in range(n_tri):
            off = 84 + i * 50
            for j in range(3):
                v = struct.unpack_from("<fff", raw, off + 12 + j * 12)
                verts.append(list(v))
    return verts


def _parse_csv_meta(raw: bytes) -> dict:
    try:
        text    = raw.decode("utf-8", errors="replace")
        reader  = csv.DictReader(io.StringIO(text))
        rows    = list(reader)
        cols    = reader.fieldnames or []
        return {"rows": len(rows), "columns": list(cols)}
    except Exception as e:
        return {"error": str(e)}


def _parse_csv_data(raw: bytes) -> dict:
    try:
        text   = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows   = list(reader)
        cols   = reader.fieldnames or []
        result = {col: [] for col in cols}
        for row in rows:
            for col in cols:
                try:
                    result[col].append(float(row[col]))
                except (ValueError, KeyError):
                    result[col].append(row.get(col, ""))
        return {"columns": list(cols), "data": result, "rows": len(rows)}
    except Exception as e:
        return {"error": str(e)}
