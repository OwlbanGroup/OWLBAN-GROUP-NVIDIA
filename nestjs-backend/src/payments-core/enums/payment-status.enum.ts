export enum PaymentStatus {
  // Initial states
  DRAFT = 'DRAFT',
  CREATED = 'CREATED',
  
  // Approval states
  PENDING_APPROVAL = 'PENDING_APPROVAL',
  APPROVED = 'APPROVED',
  REJECTED = 'REJECTED',
  
  // Submission states
  READY_TO_SUBMIT = 'READY_TO_SUBMIT',
  SUBMITTING = 'SUBMITTING',
  SUBMITTED = 'SUBMITTED',
  
  // Processing states
  PROCESSING = 'PROCESSING',
  PENDING_SETTLEMENT = 'PENDING_SETTLEMENT',
  SETTLED = 'SETTLED',
  
  // Terminal states
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
  RETURNED = 'RETURNED',
  
  // Error states
  ERROR = 'ERROR',
  TIMEOUT = 'TIMEOUT',
}
