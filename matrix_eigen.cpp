/*
 * Assignment #2 - CSS114
 * โปรแกรมหาค่าเจาะจง (Eigenvalue), เวกเตอร์เจาะจง (Eigenvector)
 * และตรวจสอบการแปลงเป็นเมทริกซ์ทแยงมุม (Diagonalization)
 * รองรับเมทริกซ์ขนาด 2x2 และ 3x3
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <string>

using namespace std;

const double EPS = 1e-9; // ค่าความคลาดเคลื่อนที่ยอมรับได้สำหรับ floating point

void printMatrix(const vector<vector<double>>& M, const string& name) {
    int n = M.size();
    cout << name << " =\n";
    for (int i = 0; i < n; i++) {
        cout << "  [ ";
        for (int j = 0; j < n; j++) {
            double val = (fabs(M[i][j]) < EPS) ? 0.0 : M[i][j]; // ป้องกัน -0.0
            cout << setw(10) << fixed << setprecision(4) << val;
            if (j < n - 1) cout << "  ";
        }
        cout << " ]\n";
    }
    cout << "\n";
}

void printVector(const vector<double>& v, const string& name) {
    cout << name << " = [ ";
    for (int i = 0; i < (int)v.size(); i++) {
        double val = (fabs(v[i]) < EPS) ? 0.0 : v[i]; // ป้องกัน -0.0
        cout << fixed << setprecision(4) << val;
        if (i < (int)v.size() - 1) cout << ",  ";
    }
    cout << " ]^T\n";
}

vector<vector<double>> matMul(const vector<vector<double>>& A,
                               const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j]; // คูณ row i ของ A กับ col j ของ B
    return C;
}

double determinant(vector<vector<double>> M) {
    int n = M.size();
    double det = 1.0;
    for (int col = 0; col < n; col++) {
        int pivot = -1;
        double maxVal = EPS;
        for (int row = col; row < n; row++) {
            if (fabs(M[row][col]) > maxVal) {
                maxVal = fabs(M[row][col]);
                pivot = row; // เลือกแถวที่มีค่า absolute มากที่สุดเป็น pivot
            }
        }
        if (pivot == -1) return 0.0; // ถ้าไม่มี pivot แสดงว่า singular det = 0
        if (pivot != col) {
            swap(M[pivot], M[col]);
            det *= -1.0; // สลับแถวทำให้เครื่องหมาย det กลับ
        }
        det *= M[col][col]; // คูณสะสม diagonal เพื่อได้ det
        for (int row = col + 1; row < n; row++) {
            double factor = M[row][col] / M[col][col]; // หา multiplier สำหรับ elimination
            for (int k = col; k < n; k++)
                M[row][k] -= factor * M[col][k]; // ลบแถวเพื่อทำให้ด้านล่าง pivot เป็น 0
        }
    }
    return det;
}

bool inverse(const vector<vector<double>>& A, vector<vector<double>>& inv) {
    int n = A.size();
    vector<vector<double>> aug(n, vector<double>(2 * n, 0.0)); // augmented matrix [A|I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = A[i][j]; // ใส่ A ฝั่งซ้าย
        aug[i][n + i] = 1.0; // ใส่ identity matrix ฝั่งขวา
    }
    for (int col = 0; col < n; col++) {
        int pivot = -1;
        double maxVal = EPS;
        for (int row = col; row < n; row++) {
            if (fabs(aug[row][col]) > maxVal) {
                maxVal = fabs(aug[row][col]);
                pivot = row; // เลือก pivot ที่มีค่า absolute มากที่สุด
            }
        }
        if (pivot == -1) return false; // ถ้าไม่มี pivot แสดงว่า singular หา inverse ไม่ได้
        swap(aug[pivot], aug[col]); // สลับแถวให้ pivot อยู่แถวปัจจุบัน
        double diag = aug[col][col];
        for (int k = 0; k < 2 * n; k++) aug[col][k] /= diag; // normalize แถว pivot
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int k = 0; k < 2 * n; k++)
                aug[row][k] -= factor * aug[col][k]; // ทำให้ทุกแถวนอกจาก pivot เป็น 0 ในคอลัมน์นั้น
        }
    }
    inv.assign(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i][j] = aug[i][n + j]; // ดึง inverse จากฝั่งขวาของ augmented matrix
    return true;
}

vector<vector<double>> nullSpace(vector<vector<double>> M) {
    int n = M.size();
    vector<int> pivotCol(n, -1); // เก็บ index คอลัมน์ของ pivot แต่ละแถว
    int row = 0;
    for (int col = 0; col < n && row < n; col++) {
        int piv = -1;
        double maxVal = EPS;
        for (int r = row; r < n; r++) {
            if (fabs(M[r][col]) > maxVal) {
                maxVal = fabs(M[r][col]);
                piv = r; // เลือก pivot
            }
        }
        if (piv == -1) continue; // ไม่มี pivot ในคอลัมน์นี้ → เป็น free variable
        swap(M[piv], M[row]); // สลับแถวให้ pivot อยู่บน
        pivotCol[row] = col; // บันทึกว่าแถวนี้ pivot อยู่คอลัมน์ไหน
        double diag = M[row][col];
        for (int k = 0; k < n; k++) M[row][k] /= diag; // normalize แถว pivot
        for (int r = 0; r < n; r++) {
            if (r == row) continue;
            double factor = M[r][col];
            for (int k = 0; k < n; k++)
                M[r][k] -= factor * M[row][k]; // elimination ทุกแถว
        }
        row++;
    }
    vector<bool> isFree(n, true);
    for (int r = 0; r < n; r++)
        if (pivotCol[r] != -1) isFree[pivotCol[r]] = false; // คอลัมน์ที่มี pivot ไม่ใช่ free
    vector<vector<double>> basis;
    for (int fc = 0; fc < n; fc++) {
        if (!isFree[fc]) continue; // ข้ามถ้าไม่ใช่ free variable
        vector<double> v(n, 0.0);
        v[fc] = 1.0; // กำหนด free variable = 1
        for (int r = 0; r < n; r++) {
            if (pivotCol[r] == -1) continue;
            v[pivotCol[r]] = -M[r][fc]; // หาค่า pivot variable จาก free variable
        }
        basis.push_back(v); // เพิ่ม eigenvector เข้า basis
    }
    return basis;
}

vector<double> normalize(const vector<double>& v) {
    double norm = 0.0;
    for (double x : v) norm += x * x; // คำนวณ squared norm
    norm = sqrt(norm); // หา Euclidean norm
    vector<double> res(v.size());
    if (norm < EPS) return v; // ถ้า norm ≈ 0 คืนค่าเดิม (zero vector)
    for (int i = 0; i < (int)v.size(); i++) res[i] = v[i] / norm; // หาร norm เพื่อ normalize
    return res;
}

vector<double> eigenvalues2x2(const vector<vector<double>>& A) {
    double tr   = A[0][0] + A[1][1];                    // trace = ผลบวก diagonal
    double det  = A[0][0]*A[1][1] - A[0][1]*A[1][0];   // determinant ของ 2x2
    double disc = tr * tr - 4.0 * det;                  // discriminant ของ char poly
    vector<double> eigs;
    if (disc < -EPS)
        eigs.push_back(NAN); // discriminant < 0 → eigenvalue เป็น complex
    else if (fabs(disc) <= EPS) {
        eigs.push_back(tr / 2.0); // discriminant = 0 → repeated eigenvalue
        eigs.push_back(tr / 2.0);
    } else {
        eigs.push_back((tr + sqrt(disc)) / 2.0); // eigenvalue ตัวที่ 1
        eigs.push_back((tr - sqrt(disc)) / 2.0); // eigenvalue ตัวที่ 2
    }
    return eigs;
}

vector<double> eigenvalues3x3(const vector<vector<double>>& A) {
    double c2 = A[0][0] + A[1][1] + A[2][2]; // trace (coefficient ของ lambda^2)
    double c1 = (A[0][0]*A[1][1] - A[0][1]*A[1][0])  // ผลบวก principal 2x2 minors
              + (A[0][0]*A[2][2] - A[0][2]*A[2][0])
              + (A[1][1]*A[2][2] - A[1][2]*A[2][1]);
    double c0 = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) // determinant ของ A
              - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
              + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    double p = c1 - c2 * c2 / 3.0;                              // coefficient p ของ depressed cubic
    double q = -2.0 * c2 * c2 * c2 / 27.0 + c1 * c2 / 3.0 - c0; // coefficient q ของ depressed cubic
    double D = -(4.0 * p * p * p + 27.0 * q * q);               // discriminant ของ depressed cubic
    double shift = c2 / 3.0; // ค่า shift สำหรับแปลงกลับจาก depressed cubic
    vector<double> eigs;
    if (D > EPS) {
        double sqrtTerm = 2.0 * sqrt(-p / 3.0);                     // amplitude ของ trig solution
        double cosArg   = (3.0 * q) / (2.0 * p) * sqrt(-3.0 / p);  // argument ของ arccos
        cosArg = max(-1.0, min(1.0, cosArg)); // clamp ป้องกัน domain error ของ acos
        double theta = acos(cosArg) / 3.0;   // มุม theta สำหรับ Viète's formula
        eigs.push_back(sqrtTerm * cos(theta)                     + shift); // root ที่ 1
        eigs.push_back(sqrtTerm * cos(theta - 2.0 * M_PI / 3.0) + shift); // root ที่ 2
        eigs.push_back(sqrtTerm * cos(theta - 4.0 * M_PI / 3.0) + shift); // root ที่ 3
    } else if (fabs(D) <= EPS) {
        if (fabs(p) < EPS && fabs(q) < EPS) {
            eigs.push_back(shift); // triple root กรณี p = q = 0
            eigs.push_back(shift);
            eigs.push_back(shift);
        } else {
            double t0 = copysign(sqrt(fabs(-p / 3.0)), q); // double root (sign ตาม q)
            double t1 = -2.0 * t0;                         // simple root
            eigs.push_back(t1 + shift); // simple root
            eigs.push_back(t0 + shift); // double root ตัวที่ 1
            eigs.push_back(t0 + shift); // double root ตัวที่ 2
        }
    } else {
        double disc     = q * q / 4.0 + p * p * p / 27.0; // discriminant ของ Cardano
        double sqrtDisc = sqrt(disc);
        double A_c = cbrt(-q / 2.0 + sqrtDisc); // Cardano term A
        double B_c = cbrt(-q / 2.0 - sqrtDisc); // Cardano term B
        eigs.push_back(A_c + B_c + shift); // real root เดียว
        eigs.push_back(NAN); // 2 complex roots ที่เหลือ
        eigs.push_back(NAN);
    }
    return eigs;
}

int main() {
    cout << "============================================\n";
    cout << "  Eigenvalue & Diagonalization Calculator\n";
    cout << "  Assignment #2 - CSS114\n";
    cout << "============================================\n\n";

    int n;
    cout << "เลือกขนาดเมทริกซ์ (2 หรือ 3): ";
    cin >> n;
    if (n != 2 && n != 3) {
        cout << "Error: รองรับเฉพาะขนาด 2x2 และ 3x3 เท่านั้น\n";
        return 1;
    }

    vector<vector<double>> A(n, vector<double>(n)); // เมทริกซ์ input
    cout << "\nกรอกค่าเมทริกซ์ A ขนาด " << n << "x" << n
         << " (แถวละ " << n << " ค่า คั่นด้วยช่องว่าง):\n";
    for (int i = 0; i < n; i++) {
        cout << "  แถว " << i + 1 << ": ";
        for (int j = 0; j < n; j++) cin >> A[i][j]; // รับค่าทีละ element
    }

    cout << "\n";
    printMatrix(A, "เมทริกซ์ A");

    cout << "--------------------------------------------\n";
    cout << " STEP 1: ค่าเจาะจง (Eigenvalues)\n";
    cout << "--------------------------------------------\n";

    vector<double> eigs = (n == 2) ? eigenvalues2x2(A) : eigenvalues3x3(A); // เลือกฟังก์ชันตามขนาด

    bool hasComplex = false;
    for (double e : eigs)
        if (isnan(e)) { hasComplex = true; break; } // ตรวจสอบว่ามี complex eigenvalue หรือไม่

    if (hasComplex) {
        cout << "* พบ Eigenvalue เป็นจำนวนเชิงซ้อน (Complex)\n";
        cout << "  => เมทริกซ์นี้ไม่สามารถแปลงเป็นเมทริกซ์ทแยงมุมได้เหนือสนาม Real\n\n";
        return 0;
    }

    for (int i = 0; i < (int)eigs.size(); i++)
        cout << "  lambda_" << i + 1 << " = " << fixed << setprecision(4) << eigs[i] << "\n";
    cout << "\n";

    cout << "--------------------------------------------\n";
    cout << " STEP 2: เวกเตอร์เจาะจง (Eigenvectors)\n";
    cout << "--------------------------------------------\n";

    vector<double> uniqueEigs; // เก็บ eigenvalue ที่ไม่ซ้ำกัน
    for (double e : eigs) {
        bool found = false;
        for (double ue : uniqueEigs)
            if (fabs(e - ue) < EPS * 10) { found = true; break; } // เช็คว่าซ้ำหรือยัง
        if (!found) uniqueEigs.push_back(e);
    }

    vector<vector<double>> eigVecs; // เก็บ eigenvectors ทั้งหมด (จะเป็น columns ของ P)
    for (double lambda : uniqueEigs) {
        cout << "  lambda = " << fixed << setprecision(4) << lambda << ":\n";
        vector<vector<double>> AminLI(n, vector<double>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                AminLI[i][j] = A[i][j] - (i == j ? lambda : 0.0); // คำนวณ (A - lambda*I)
        vector<vector<double>> basis = nullSpace(AminLI); // หา null space ของ (A - lambda*I)
        if (basis.empty()) {
            cout << "    ไม่พบ eigenvector (เกิดข้อผิดพลาด)\n";
        } else {
            for (int k = 0; k < (int)basis.size(); k++) {
                vector<double> nv = normalize(basis[k]); // normalize eigenvector
                eigVecs.push_back(nv);
                cout << "    v_" << eigVecs.size() << " = [ ";
                for (int i = 0; i < n; i++) {
                    double val = (fabs(nv[i]) < EPS) ? 0.0 : nv[i]; // ป้องกัน -0.0
                    cout << fixed << setprecision(4) << val;
                    if (i < n - 1) cout << ",  ";
                }
                cout << " ]^T  (normalized)\n";
            }
        }
    }
    cout << "\n";

    cout << "--------------------------------------------\n";
    cout << " STEP 3: ตรวจสอบการ Diagonalize\n";
    cout << "--------------------------------------------\n";

    bool isDiagonalizable = ((int)eigVecs.size() == n); // diagonalizable ก็ต่อเมื่อมี n eigenvectors

    if (!isDiagonalizable) {
        cout << "* ผลการตรวจสอบ: ไม่สามารถแปลงเป็นเมทริกซ์ทแยงมุมได้\n\n";
        cout << "  เหตุผล:\n";
        cout << "  - เมทริกซ์ขนาด " << n << "x" << n
             << " ต้องการ " << n << " eigenvectors ที่อิสระเชิงเส้น\n";
        cout << "  - แต่หาได้เพียง " << eigVecs.size() << " eigenvector\n";
        cout << "  - นั่นหมายความว่ามี eigenvalue ซ้ำที่มี geometric multiplicity\n";
        cout << "    น้อยกว่า algebraic multiplicity (eigenvectors ไม่เพียงพอ)\n";
        cout << "  - ดังนั้น เมทริกซ์นี้เป็น defective matrix และไม่ diagonalizable\n";
        return 0;
    }

    cout << "* ผลการตรวจสอบ: สามารถแปลงเป็นเมทริกซ์ทแยงมุมได้\n";
    cout << "  (มี " << n << " eigenvectors อิสระเชิงเส้นครบถ้วน)\n\n";

    cout << "--------------------------------------------\n";
    cout << " STEP 4: เมทริกซ์ P, P^-1 และ D\n";
    cout << "--------------------------------------------\n";

    vector<vector<double>> P(n, vector<double>(n));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            P[i][j] = eigVecs[j][i]; // แต่ละคอลัมน์ของ P คือ eigenvector

    vector<vector<double>> D(n, vector<double>(n, 0.0)); // เมทริกซ์ทแยงมุม D
    for (int j = 0; j < n; j++) {
        vector<double> Av(n, 0.0);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                Av[i] += A[i][k] * eigVecs[j][k]; // คำนวณ A * v_j
        for (int i = 0; i < n; i++) {
            if (fabs(eigVecs[j][i]) > EPS) {
                D[j][j] = Av[i] / eigVecs[j][i]; // lambda = (Av)[i] / v[i]
                break;
            }
        }
    }

    vector<vector<double>> Pinv;
    if (!inverse(P, Pinv)) {
        cout << "Error: ไม่สามารถหา inverse ของ P ได้ (P เป็น singular)\n";
        return 1;
    }

    printMatrix(P, "P (คอลัมน์คือ eigenvectors)");
    printMatrix(Pinv, "P^-1");
    printMatrix(D, "D = P^-1 * A * P (เมทริกซ์ทแยงมุม)");

    cout << "--------------------------------------------\n";
    cout << " การตรวจสอบ: P^-1 * A * P ควรเท่ากับ D\n";
    cout << "--------------------------------------------\n";

    vector<vector<double>> check = matMul(matMul(Pinv, A), P); // คำนวณ P^-1 * A * P
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (fabs(check[i][j]) < EPS) check[i][j] = 0.0; // ทำความสะอาดค่า near-zero

    printMatrix(check, "P^-1 * A * P (ผลตรวจสอบ)");

    bool verified = true;
    for (int i = 0; i < n && verified; i++)
        for (int j = 0; j < n && verified; j++)
            if (fabs(check[i][j] - D[i][j]) > 1e-6) verified = false; // เปรียบเทียบกับ D

    if (verified)
        cout << "✓ ตรวจสอบผ่าน: P^-1 * A * P = D อย่างถูกต้อง\n\n";
    else
        cout << "✗ ตรวจสอบไม่ผ่าน: อาจเกิดจาก numerical error\n\n";

    cout << "============================================\n";
    cout << "  การคำนวณเสร็จสมบูรณ์\n";
    cout << "============================================\n";

    return 0;
}
