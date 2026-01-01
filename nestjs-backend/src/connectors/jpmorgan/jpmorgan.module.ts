import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { JpmorganService } from './jpmorgan.service';
import { JpmorganTokenService } from './jpmorgan-token.service';
import { JpmorganController } from './jpmorgan.controller';

@Module({
  imports: [
    HttpModule.register({
      timeout: 5000,
    }),
  ],
  controllers: [JpmorganController],
  providers: [JpmorganTokenService, JpmorganService],
  exports: [JpmorganService, JpmorganTokenService],
})
export class JpmorganModule {}
