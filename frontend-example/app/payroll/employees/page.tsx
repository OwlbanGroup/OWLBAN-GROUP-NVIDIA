import { apiGet } from '../../../lib/api';
import { EmployeeForm } from '../../../components/Payroll/EmployeeForm';
import { EmployeeTable } from '../../../components/Payroll/EmployeeTable';

const ORG_ID = 'ORG_UUID_HERE';

async function getEmployees() {
  return apiGet<any[]>(`/payroll/employees/${ORG_ID}`);
}

export default async function EmployeesPage() {
  const employees = await getEmployees();

  return (
    <div>
      <EmployeeForm orgId={ORG_ID} onCreated={() => window.location.reload()} />
      <EmployeeTable employees={employees} />
    </div>
  );
}
